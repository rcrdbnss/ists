import numpy as np
import tensorflow as tf
from ists_.model.encoder import GlobalSelfAttention, CrossAttention


class GlobalWindowAttention(GlobalSelfAttention):
    """Masked global attention: full (C*S)x(C*S) scores, window enforced by mask.

    Input (B, C, S, E) -> output (B, C, S, E). Mask: (B, C, S).
    """

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number.")
        self.window_size = window_size

    def build(self, input_shape):
        _, C, S, E = input_shape
        self.channels, self.sequence_length, self.d_model = C, S, E

        idx = np.arange(S)
        local_mask = np.abs(idx[:, None] - idx[None, :]) <= self.window_size // 2
        # Same window for every channel pair: (C*S, C*S), kept boolean.

        with tf.init_scope():
            # Expand to (C*S, C*S) for multi-channel input
            self.local_mask = tf.constant(np.tile(local_mask, [C, C]))  # (C*S, C*S)

        super().build(input_shape)

    def call(self, x, attention_mask=None):  # x: (B, C, S, E) attn_mask: (B, C, S)
        C, S, E = self.channels, self.sequence_length, self.d_model
        x = tf.reshape(x, [-1, C*S, E])

        local_mask = self.local_mask[tf.newaxis, :, :]  # (1, C*S, C*S)
        local_mask = tf.cast(local_mask, self.compute_dtype)

        # 2. Combine with any external mask (e.g., padding mask) if provided
        if attention_mask is not None:
            # Keras MHA masks are boolean. False means "mask this token".
            # We combine them with a logical AND. A token is attended to
            # only if it's valid in BOTH the local window AND the external mask.
            # The external mask needs to be reshaped to be broadcastable with the local mask.
            # External mask shape: (B, 1, C*S)
            # local_mask shape: (1, C*S, C*S)
            # Combined shape will be broadcast to (B, C*S, C*S)
            attention_mask = tf.reshape(attention_mask, [-1, 1, C*S])
            local_mask = local_mask * attention_mask  # (B, C*S, C*S)

        # 3. Call the parent's call method with the final mask
        x = super().call(x, attention_mask=local_mask)
        return tf.reshape(x, [-1, C, S, E])


class GlobalWindowAttentionV2(CrossAttention):
    """Windowed attention: each token attends to all channels within its time
    window. Scores scale as O(S * W * C^2) instead of O(S^2 * C^2).

    Input (B, C, S, E) -> output (B, C, S, E). Mask: (B, C, S).
    """

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number.")
        self.window_size = window_size

    def build(self, input_shape):
        _, C, S, E = input_shape
        self.channels, self.sequence_length, self.d_model = C, S, E
        W = self.window_size
        # Validity of each window slot at each timestep: position s sees
        # s + w - W//2, which must lie in [0, S). Used when attention_mask
        # is None to hide extract_patches' zero padding at the edges.
        k = np.arange(S)[:, None] + np.arange(W)[None, :] - W // 2  # (S, W)
        edge = (k >= 0) & (k < S)
        edge = np.repeat(edge[:, :, None], C, axis=2)  # (S, W, C)
        with tf.init_scope():
            self.edge_mask = tf.constant(edge.reshape(S, W * C))
        super().build(input_shape)

    def windowing(self, x):
        x_4d = tf.expand_dims(x, axis=1)  # (B, 1, S, E)
        patches = tf.image.extract_patches(
            images=x_4d, sizes=[1, 1, self.window_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        return tf.reshape(patches, [-1, self.sequence_length, self.window_size, tf.shape(x)[-1]])

    def call(self, x, attention_mask=None):  # x: (B, C, S, E) attn_mask: (B, C, S)
        C, S, W, E = self.channels, self.sequence_length, self.window_size, self.d_model

        # Time-local windows per channel: (B*C, S, W, E)
        win = self.windowing(tf.reshape(x, [-1, S, E]))
        win = tf.reshape(win, [-1, C, S, W, E])
        # Group windows by timestep, (w, c)-ordered: (B, S, W, C, E)
        win = tf.transpose(win, [0, 2, 3, 1, 4])
        # One shared context per (batch, timestep) — no per-channel tiling.
        context = tf.reshape(win, [-1, W * C, E])

        # All C queries of a timestep form one attention "batch element".
        q = tf.transpose(x, [0, 2, 1, 3])  # (B, S, C, E)
        q = tf.reshape(q, [-1, C, E])

        if attention_mask is None:
            # Only the sequence edges need masking (zero padding from extract_patches).
            # Broadcast the static edge mask over B.
            m = tf.broadcast_to(self.edge_mask[tf.newaxis, :, :], [-1, S, W * C])
            mha_mask = tf.reshape(m, [-1, 1, W * C])
        else:
            m = tf.reshape(tf.cast(attention_mask, self.compute_dtype), [-1, S, 1])
            mw = self.windowing(m)  # (B*C, S, W, 1)
            mw = tf.reshape(mw, [-1, C, S, W])
            mw = tf.transpose(mw, [0, 2, 3, 1])  # (B, S, W, C)
            mha_mask = tf.reshape(mw, [-1, 1, W * C])  # broadcasts over C queries

        out = super().call(x=q, context=context, attention_mask=mha_mask)  # (B*S, C, E)

        out = tf.reshape(out, [-1, S, C, E])
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, C, S, E)
        return out


if __name__ == "__main__":
    import random

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    B, C, S, E, W = 1, 2, 8, 16, 3
    x = tf.random.normal([B, C, S, E])

    num_heads = 1
    kwargs = {
        'num_heads': num_heads,
        'key_dim': E // num_heads,
        'dropout': 0.0,
        'kernel_regularizer': None,
    }

    # Create a padding mask all-ones for this test (no padding)
    attn_mask_global = tf.ones([B, C, S], dtype=tf.bool)

    # Run V1 (masked global)
    tf.random.set_seed(seed)  # reset seed to get same initialization
    np.random.seed(seed)
    random.seed(seed)

    v1_layer = GlobalWindowAttention(window_size=W, **kwargs)
    out1 = v1_layer(x, attention_mask=attn_mask_global)

    # Run V2 (patched)
    tf.random.set_seed(seed)  # reset seed to get same initialization
    np.random.seed(seed)
    random.seed(seed)

    v2_layer = GlobalWindowAttentionV2(window_size=W, **kwargs)
    out2 = v2_layer(x, attention_mask=attn_mask_global)

    # Compare
    diff = tf.reduce_max(tf.abs(out1 - out2))
    print("max abs diff:", diff.numpy())
