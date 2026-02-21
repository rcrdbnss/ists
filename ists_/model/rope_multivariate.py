import tensorflow as tf

from ists_.model.rope import get_rotary_matrix, RoPEMultiHeadAttention, apply_rotary_pos_emb, GlobalSelfAttention


class RoPEMVMultiHeadAttention(RoPEMultiHeadAttention):
    """
    Custom MultiHeadAttention with RoPE, Dropout, and Kernel Regularization.
    Expects input shape as (batch, channels, seq_len, d_model), rotates each channel independently,
    then reshapes to (batch, channels*seq_len, d_model) for attention computation.
    """

    def build(self, input_shape):
        _, channels, seq_len, _ = input_shape
        self.channels = channels
        self.freqs = get_rotary_matrix(max_len=seq_len, d_model=self.depth, max_freq=self.max_freq)

    def call(self, v, k, q, mask=None):  # x: (B, C, S, D), mask: (B, 1, C*S)
        batch_size = tf.shape(q)[0]

        # reshape to 3D
        q = tf.reshape(q, (batch_size * self.channels, -1, self.d_model))  # (B*C, S, D)
        k = tf.reshape(k, (batch_size * self.channels, -1, self.d_model))  # (B*C, S, D)
        v = tf.reshape(v, (batch_size * self.channels, -1, self.d_model))  # (B*C, S, D)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size*self.channels)  # (B*C, num_heads, S, depth)
        k = self.split_heads(k, batch_size*self.channels)  # (B*C, num_heads, S, depth)
        v = self.split_heads(v, batch_size*self.channels)  # (B*C, num_heads, S, depth)

        # Apply RoPE
        q = apply_rotary_pos_emb(q, self.freqs)
        k = apply_rotary_pos_emb(k, self.freqs)

        # reshape, transpose and reshape to 4D
        q = tf.reshape(q, (batch_size, self.channels, self.num_heads, -1, self.depth))  # (B, C, num_heads, S, depth)
        k = tf.reshape(k, (batch_size, self.channels, self.num_heads, -1, self.depth))  # (B, C, num_heads, S, depth)
        v = tf.reshape(v, (batch_size, self.channels, self.num_heads, -1, self.depth))  # (B, C, num_heads, S, depth)
        q = tf.transpose(q, perm=[0, 2, 1, 3, 4])  # (B, num_heads, C, S, depth)
        k = tf.transpose(k, perm=[0, 2, 1, 3, 4])  # (B, num_heads, C, S, depth)
        v = tf.transpose(v, perm=[0, 2, 1, 3, 4])  # (B, num_heads, C, S, depth)
        q = tf.reshape(q, (batch_size, self.num_heads, -1, self.depth))  # (B, num_heads, C*S, depth)
        k = tf.reshape(k, (batch_size, self.num_heads, -1, self.depth))  # (B, num_heads, C*S, depth)
        v = tf.reshape(v, (batch_size, self.num_heads, -1, self.depth))  # (B, num_heads, C*S, depth)

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B, num_heads, C*S, C*S)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            mask = mask[:, tf.newaxis, :]
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply Dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)  # (B, num_heads, C*S, depth)
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (B, C*S, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))  # (B, C*S, D)

        # reshape to input shape
        concat_attention = tf.reshape(concat_attention, (batch_size, self.channels, -1, self.d_model))  # (B, C, S, D)

        return self.dense(concat_attention)


class GlobalMVSelfAttention(GlobalSelfAttention):
    def __init__(self, num_heads, key_dim, dropout=0.1, kernel_regularizer=None, pre_layernorm=False, rms_scaling=False,
                 max_freq=10000.0, **kwargs):
        super().__init__(num_heads=num_heads, key_dim=key_dim, dropout=dropout, kernel_regularizer=kernel_regularizer,
                         pre_layernorm=pre_layernorm, rms_scaling=rms_scaling, max_freq=max_freq, **kwargs)
        # replace the standard MHA with our custom MV version
        self.mha = RoPEMVMultiHeadAttention(
            d_model=key_dim,
            num_heads=num_heads,
            dropout_rate=dropout,
            kernel_regularizer=kernel_regularizer,
            max_freq=self.max_freq,
        )
