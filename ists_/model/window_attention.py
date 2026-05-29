import tensorflow as tf
from ists_.model.encoder import GlobalSelfAttention, CrossAttention


# def create_sliding_window_mask(sequence_length, window_size):
#     """Creates a boolean sliding window attention mask."""
#     if window_size % 2 == 0:
#         raise ValueError("Window size must be an odd number.")
#     q_indices = tf.range(sequence_length)
#     k_indices = tf.range(sequence_length)
#     distance_matrix = q_indices[:, tf.newaxis] - k_indices[tf.newaxis, :]
#     half_window = window_size // 2
#     return tf.abs(distance_matrix) <= half_window


class GlobalWindowAttention(GlobalSelfAttention):
    """
    Specializes GlobalSelfAttention by applying a sliding-window mask.
    """

    def __init__(self, sequence_length, window_size, channels=1, **kwargs):
        # Initialize the parent (GlobalSelfAttention)
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.channels = channels

    def create_sliding_window_mask(self):
        """Creates a boolean sliding window attention mask."""
        S = self.sequence_length
        W = self.window_size
        if W % 2 == 0:
            raise ValueError("Window size must be an odd number.")
        q_indices = tf.range(S)
        k_indices = tf.range(S)
        distance_matrix = q_indices[:, tf.newaxis] - k_indices[tf.newaxis, :]
        half_window = W // 2
        return tf.abs(distance_matrix) <= half_window

    def build(self, input_shape):
        C = self.channels
        # S = self.sequence_length

        with tf.init_scope():
            # 1. Create the local sliding-window mask
            # local_mask = create_sliding_window_mask(S, self.window_size)  # (S, S)
            local_mask = self.create_sliding_window_mask()  # (S, S)

            """# KMask+GAnoCLS: CLS does not attend any token
            is_not_first_token = tf.range(S) > 0
            row_mask = is_not_first_token[:, tf.newaxis]
            col_mask = is_not_first_token[tf.newaxis, :]
            local_mask = tf.logical_and(local_mask, tf.logical_and(row_mask, col_mask))"""

            # Expand to (C*S, C*S) for multi-channel input
            local_mask = tf.tile(local_mask, [C, C])  # (C*S, C*S)

            """# KMask+GAwithCLS: CLS attends all tokens of its own channel
            channel_indices = tf.range(S*C) // S
            query_is_cls = tf.range(S*C) % S == 0  # (C*S,)
            query_is_cls = query_is_cls[:, tf.newaxis]
            same_channel = (channel_indices[:, tf.newaxis] == channel_indices[tf.newaxis, :])
            reactivation_mask = tf.logical_and(query_is_cls, same_channel)
            structural_mask = tf.logical_or(local_mask, reactivation_mask)
            local_mask = structural_mask"""

            self.local_mask = tf.Variable(initial_value=tf.cast(local_mask, tf.float32),
                                          trainable=False, dtype=tf.float32, name="local_mask")

        super().build(input_shape)

    def call(self, x, attention_mask=None):  # x: (B, C*S, E) attn_mask: (B, 1, C*S)
        local_mask = tf.cast(self.local_mask, self.compute_dtype)

        # 2. Combine with any external mask (e.g., padding mask) if provided
        if attention_mask is not None:
            # Keras MHA masks are boolean. False means "mask this token".
            # We combine them with a logical AND. A token is attended to
            # only if it's valid in BOTH the local window AND the external mask.
            # The external mask needs to be reshaped to be broadcastable with the local mask.
            # External mask shape: (B, 1, C*S)
            # local_mask shape: (C*S, C*S) -> (1, C*S, C*S)
            # Combined shape will be broadcast to (B, C*S, C*S)
            combined_mask = local_mask[tf.newaxis, :, :] * attention_mask  # (B, C*S, C*S)
        else:
            combined_mask = local_mask

        # 3. Call the parent's call method with the final mask
        return super().call(x, attention_mask=combined_mask)


class GlobalWindowAttentionV2(CrossAttention):
    """
    Specializes CrossAttention by applying a sliding-window mask.
    """

    def __init__(self, sequence_length, window_size, channels=1, **kwargs):
        # Initialize the parent (CrossAttention)
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.channels = channels

    def windowing(self, x):
        x_4d = tf.expand_dims(x, axis=1)  # (B, 1, S, E)
        patches = tf.image.extract_patches(
            images=x_4d, sizes=[1, 1, self.window_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        B, S, E = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        return tf.reshape(patches, [B, S, self.window_size, E])  # (B, S, W, E)

    def call(self, x, attention_mask=None):  # x: (B, C*S, E) attn_mask: (B, 1, C*S)
        C = self.channels
        S = self.sequence_length
        W = self.window_size

        x_shape = tf.shape(x)
        B, _, E = x_shape[0], x_shape[1], x_shape[2]
        x = tf.reshape(x, (B, C, S, E))
        attn_mask = tf.reshape(attention_mask, (B, C, S))

        # 1. Create time-local windows for each channel independently
        # Reshape to (B*C, S, E) to apply windowing
        x_bc_reshaped = tf.reshape(x, [B * C, S, E])
        # Create time-local windows -> (B*C, S, W, E)
        time_windows = self.windowing(x_bc_reshaped)
        # Reshape back to include channel dimension -> (B, C, S, W, E)
        time_windows = tf.reshape(time_windows, [B, C, S, W, E])
        # Transpose to (B, S, C, W, E) for easier assembly
        time_windows = tf.transpose(time_windows, [0, 2, 1, 3, 4])

        # 2. Assemble the cross-channel context for each timestep
        # For each timestep 't', the context is the windows from ALL channels.
        # Permute to group windows by timestep: (B, S, W, C, E)
        time_windows = tf.transpose(time_windows, [0, 1, 3, 2, 4])
        # Reshape into a single context sequence per timestep: (B, S, W*C, E)
        context_per_ts = tf.reshape(time_windows, [B, S, W * C, E])

        # 3. Broadcast the context to all channels for each timestep
        # We need each of the 'C' tokens at timestep 't' to see the same context.
        # Tile along a new channel axis: (B, S, 1, W*C, E) -> (B, S, C, W*C, E)
        context_tiled = tf.tile(context_per_ts[:, :, tf.newaxis, :, :], [1, 1, C, 1, 1])

        # 4. Apply the "batching trick"
        # Reshape query and context for the parent call
        x = tf.transpose(x, [0, 2, 1, 3])  # (B, C, S, E) -> (B, S, C, E)
        query_reshaped = tf.reshape(x, [B * S * C, 1, E])  # (B, S, C, E) -> (B*S*C, 1, E)
        context_reshaped = tf.reshape(context_tiled, [B * S * C, W * C, E])  # (B, S, C, W*C, E) -> (B*S*C, W*C, E)

        # 5. Handle the attention mask

        # KMask+GAnoCLS: CLS does not attend any token
        is_not_first_token = tf.cast(tf.range(S) > 0, dtype=tf.float32)
        is_not_first_token = is_not_first_token[tf.newaxis, tf.newaxis, :]
        attn_mask = attn_mask * is_not_first_token  # (B, C, S)

        # Apply the same windowing and tiling procedure to the mask
        mask_bc_reshaped = tf.reshape(attn_mask, [B * C, S, 1])

        windowed_mask = self.windowing(tf.cast(mask_bc_reshaped, tf.float32))
        windowed_mask = tf.reshape(windowed_mask, [B, C, S, W])
        windowed_mask = tf.transpose(windowed_mask, [0, 2, 1, 3])  # (B, S, C, W)

        windowed_mask = tf.transpose(windowed_mask, [0, 1, 3, 2])  # (B, S, W, C)
        mask_context_per_ts = tf.reshape(windowed_mask, [B, S, W * C])

        mask_tiled = tf.tile(mask_context_per_ts[:, :, tf.newaxis, :], [1, 1, C, 1])  # (B, S, C, W*C)

        # KMask+GAnoCLS: CLS does not attend any token
        # Workaround for: mask_tiled[:, 0] = 0
        is_not_first_token = tf.cast(tf.range(S) > 0, dtype=tf.float32)
        is_not_first_token = is_not_first_token[tf.newaxis, :, tf.newaxis, tf.newaxis]
        mask_tiled = mask_tiled * is_not_first_token  # (B, S, C, W*C)

        final_mha_mask = tf.reshape(mask_tiled, [B * S * C, 1, W * C])  # (B*S*C, 1, W*C)

        # Call the parent CrossAttention method
        attn_output_reshaped = super().call(
            x=query_reshaped,
            context=context_reshaped,
            attention_mask=final_mha_mask
        )  # (B*S*C, 1, E)

        # Reshape back to (B, C, S, E)
        attn_output = tf.reshape(attn_output_reshaped, [B, S, C, E])
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])  # (B, C, S, E) -> (B, C, S, E)
        attn_output = tf.reshape(attn_output, [B, C * S, E])  # (B, C, S, E) -> (B, C*S, E)
        return attn_output


if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    import random

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    B, C, S, E, W = 1, 2, 8, 16, 3
    x = tf.random.normal([B, C * S, E])

    kwargs = {
        'num_heads': 1,
        'key_dim': E,
        'dropout': 0.0,
        'kernel_regularizer': None,
    }

    # Create a padding mask all-ones for this test (no padding)
    attn_mask_global = tf.ones([B, 1, C * S], dtype=tf.bool)

    # Run V1 (masked global)
    tf.random.set_seed(seed)  # reset seed to get same initialization
    np.random.seed(seed)
    random.seed(seed)

    v1_layer = GlobalWindowAttention(sequence_length=S, window_size=W, channels=C, **kwargs)
    out1 = v1_layer(x, attention_mask=attn_mask_global)

    # Run V2 (patched)
    tf.random.set_seed(seed)  # reset seed to get same initialization
    np.random.seed(seed)
    random.seed(seed)

    v2_layer = GlobalWindowAttentionV2(sequence_length=S, window_size=W, channels=C, **kwargs)
    out2 = v2_layer(x, attention_mask=attn_mask_global)

    # Compare
    diff = tf.reduce_max(tf.abs(out1 - out2))
    print("max abs diff:", diff.numpy())
