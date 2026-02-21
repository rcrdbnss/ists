import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, rms_scaling=False, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization(rms_scaling=rms_scaling)
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    last_attn_scores = None

    def __init__(self, pre_layernorm=False, rms_scaling=False, **kwargs):
        super().__init__(rms_scaling=rms_scaling, **kwargs)
        self.pre_layernorm = pre_layernorm

    def call(self, x, context, attention_mask=None):
        y = x
        if self.pre_layernorm: y = self.layernorm(y)
        attn_output, attn_scores = self.mha(
            query=y,
            key=context,
            value=context,
            return_attention_scores=True,
            attention_mask=attention_mask
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        if not self.pre_layernorm: x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    last_attn_scores = None

    def __init__(self, pre_layernorm=False, rms_scaling=False, **kwargs):
        super().__init__(rms_scaling=rms_scaling, **kwargs)
        self.pre_layernorm = pre_layernorm

    def call(self, x, attention_mask=None):
        y = x
        if self.pre_layernorm: y = self.layernorm(y)
        attn_output, attn_scores = self.mha(
            query=y,
            key=y,
            value=y,
            return_attention_scores=True,
            attention_mask=attention_mask
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        if not self.pre_layernorm: x = self.layernorm(x)

        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, activation='relu', dropout_rate=0.1, kernel_regularizer=None,
                 pre_layernorm=False, rms_scaling=False):
        super().__init__()

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation, kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(rms_scaling=rms_scaling)
        self.pre_layernorm = pre_layernorm

    def call(self, x, **kwargs):
        y = x
        if self.pre_layernorm: y = self.layer_norm(y)
        x = self.add([x, self.seq(y)])
        if not self.pre_layernorm: x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None,
                 pre_layernorm=False, rms_scaling=False, **kwargs):
        super().__init__()

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            **reg,
            pre_layernorm=pre_layernorm, rms_scaling=rms_scaling
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate, **reg,
            pre_layernorm=pre_layernorm, rms_scaling=rms_scaling
        )
        self.last_attn_scores = None

    def call(self, x, attention_mask=None):
        x = self.self_attention(x, attention_mask)
        self.last_attn_scores = self.self_attention.last_attn_scores
        x = self.ffn(x)

        return x


class MVEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None,
                 pre_layernorm=False, rms_scaling=False):
        super().__init__()

        self.encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            pre_layernorm=pre_layernorm, rms_scaling=rms_scaling
        )

    def call(self, x, attn_mask=None):  # x: (v, b, t, e) attn_mask: (v, b, t)
        shape = tf.shape(x)
        v, b, t, e = shape[0], shape[1], shape[2], shape[3]

        if attn_mask is None:
            attn_mask = tf.ones((v, b, t), dtype=tf.float32)

        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (v, b, t, e) -> (b, v, t, e)
        attn_mask = tf.transpose(attn_mask, perm=[1, 0, 2])  # attn_mask: (v, b, t) -> (b, v, t)

        x = tf.reshape(x, (b, v * t, e))  # x: (b, v*t, e)
        attn_mask = tf.reshape(attn_mask, (b, v * t, 1))  # attn_mask: (b, v*t)
        x = self.encoder(x, attn_mask)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (b, v, t, e) -> (v, b, t, e)
        return x


if __name__ == '__main__':
    print("Hello, world!")
