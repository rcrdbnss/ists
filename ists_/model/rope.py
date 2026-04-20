import numpy as np
import tensorflow as tf

from ists_.model.encoder import FeedForward

'''def get_rotary_matrix(max_len, d_model, max_freq=10000.0):
    """Generates the RoPE frequency embedding."""
    pos = tf.range(max_len, dtype=tf.float32)
    inv_freq = 1.0 / (max_freq ** (tf.range(0, d_model, 2, dtype=tf.float32) / d_model))
    freqs = tf.einsum('i,j->ij', pos, inv_freq)
    emb = tf.concat((freqs, freqs), axis=-1)
    return emb[tf.newaxis, tf.newaxis, :, :]'''

def get_rotary_matrix(max_len, d_model, max_freq=10000.0):
    """Generates the RoPE frequency embedding."""
    pos = np.arange(max_len, dtype=float)
    inv_freq = 1.0 / (max_freq ** (np.arange(0, d_model, 2) / d_model))
    freqs = np.einsum('i,j->ij', pos, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    return emb[np.newaxis, np.newaxis, :, :]


def rotate_half(x):
    """Rotates the last dimension of the input tensor."""
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(x, freqs):
    """Applies the Rotary Positional Embedding."""
    '''seq_len = tf.shape(x)[2]
    freqs = freqs[:, :, :seq_len, :]
    cos = tf.cos(freqs)
    sin = tf.sin(freqs)'''
    cos, sin = tf.cos(tf.cast(freqs, tf.float32)), tf.sin(tf.cast(freqs, tf.float32))
    return (x * cos) + (rotate_half(x) * sin)


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return x


class RoPEMultiHeadAttention(tf.keras.layers.Layer):
    """
    Custom MultiHeadAttention with RoPE, Dropout, and Kernel Regularization.
    """

    def __init__(self, d_model, num_heads, dropout_rate=0.0, kernel_regularizer=None, max_freq=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_freq = max_freq

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // self.num_heads

        # Apply kernel_regularizer to all dense layers
        self.wq = tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer)
        self.wk = tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer)
        self.wv = tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer)
        self.dense = tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer)

        # Dropout layer for attention weights
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # self.freqs = get_rotary_matrix(max_len=2048, d_model=self.depth, max_freq=self.max_freq)

    def build(self, input_shape):
        _, seq_len, _ = input_shape
        self.freqs = get_rotary_matrix(max_len=seq_len, d_model=self.depth, max_freq=self.max_freq)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply RoPE
        q = apply_rotary_pos_emb(q, self.freqs)
        k = apply_rotary_pos_emb(k, self.freqs)

        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            # mask = mask[:, tf.newaxis, tf.newaxis, :]
            mask = mask[:, tf.newaxis, :]
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply Dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(concat_attention)


class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.1, kernel_regularizer=None, pre_layernorm=False, rms_scaling=False,
                 max_freq=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.pre_layernorm = pre_layernorm
        self.rms_scaling = rms_scaling
        self.max_freq = max_freq

        self.mha = RoPEMultiHeadAttention(
            d_model=key_dim,
            num_heads=num_heads,
            dropout_rate=dropout,
            kernel_regularizer=kernel_regularizer,
            max_freq=self.max_freq,
        )
        self.layernorm = tf.keras.layers.RMSNormalization() if self.rms_scaling else tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.add = tf.keras.layers.Add()

    def call(self, x, mask=None):
        y = x
        if self.pre_layernorm: y = self.layernorm(y)
        attn_output = self.mha(q=y, k=y, v=y, mask=mask)
        x = self.add([x, attn_output])
        if not self.pre_layernorm: x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, activation='relu', dropout_rate=0.1, kernel_regularizer=None, pre_layernorm=False,
                 rms_scaling=False, **kwargs):
        super().__init__(**kwargs)
        self.pre_layernorm = pre_layernorm
        self.rms_scaling = rms_scaling

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation, kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(rms_scaling=self.rms_scaling)

    def call(self, x, **kwargs):
        y = x
        if self.pre_layernorm: y = self.layer_norm(y)
        x = self.add([x, self.seq(y)])
        if not self.pre_layernorm: x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, kernel_regularizer=None, pre_layernorm=False,
                 rms_scaling=False, max_freq=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.pre_layernorm = pre_layernorm
        self.rms_scaling = rms_scaling
        self.max_freq = max_freq

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            kernel_regularizer=kernel_regularizer,
            pre_layernorm=self.pre_layernorm,
            rms_scaling=self.rms_scaling,
            max_freq=self.max_freq,
            **kwargs
        )

        self.ffn = FeedForward(
            d_model, dff, dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling, **kwargs,
        )

    def call(self, x, mask=None):
        x = self.self_attention(x, mask=mask)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1, kernel_regularizer=None,
                 pre_layernorm=False, rms_scaling=False, max_freq=10000.0, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = TokenEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            kernel_regularizer=kernel_regularizer,
            pre_layernorm=pre_layernorm,
            rms_scaling=rms_scaling,
            max_freq=max_freq,
            **kwargs,
        ) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        mask = self.pos_embedding.compute_mask(x)
        x = self.pos_embedding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask)

        return x
