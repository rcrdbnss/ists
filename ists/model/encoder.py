import numpy as np
import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    last_attn_scores = None

    # noinspection PyMethodOverriding
    def call(self, x, context, **kwargs):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    last_attn_scores = None

    def call(self, x, **kwargs):
        attn_output, attn_scores = self.mha(
            query=x,
            key=x,
            value=x,
            return_attention_scores=True
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, **kwargs):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)
        self.last_attn_scores = None

    def call(self, x, **kwargs):
        x = self.self_attention(x)
        self.last_attn_scores = self.self_attention.last_attn_scores
        x = self.ffn(x)

        return x


class CrossEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(CrossEncoderLayer, self).__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)
        self.last_attn_scores = None

    # noinspection PyMethodOverriding
    def call(self, x, context, **kwargs):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class GlobalEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(GlobalEncoderLayer, self).__init__()

        self.temporal_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )
        self.spatial_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

        self.exogenous_encoder = CrossEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

        self.global_encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

        self.last_attn_scores = None

    # noinspection PyMethodOverriding
    def call(self, time_x, exogenous_x, spatial_x, **kwargs):
        # Compute temporal, spatial, and exogenous embedding
        time_x = self.temporal_encoder(x=time_x)
        spatial_x = self.spatial_encoder(x=spatial_x)
        exogenous_x = self.exogenous_encoder(x=exogenous_x, context=time_x)

        # Concatenate along T dimension
        t1 = time_x.shape[1]
        t2 = spatial_x.shape[1]
        # t3 = exogenous_x.shape[1]
        embedded_x = tf.concat([time_x, spatial_x, exogenous_x], axis=1)

        # Compute global cross attention
        embedded_x = self.global_encoder(embedded_x)

        # Divide the tensor back into three separate tensors
        time_x = embedded_x[:, :t1, :]
        spatial_x = embedded_x[:, t1:t1 + t2, :]
        exogenous_x = embedded_x[:, t1 + t2:, :]

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.global_encoder.last_attn_scores

        return time_x, exogenous_x, spatial_x


if __name__ == '__main__':
    # Input
    data1 = np.random.randn(100, 24, 128).astype(np.float32)
    data2 = np.random.randn(100, 12, 128).astype(np.float32)
    data3 = np.random.randn(100, 6, 128).astype(np.float32)

    # # EncoderLayer init & call
    # encoder = EncoderLayer(d_model=128, num_heads=2, dff=256, dropout_rate=0.1)
    # emb1 = encoder(data1)
    # print(f'EncoderLayer: {data1.shape} {emb1.shape}')
    #
    # # CrossEncoderLayer init & call
    # cross_encoder = CrossEncoderLayer(d_model=128, num_heads=2, dff=256, dropout_rate=0.1)
    # emb2 = cross_encoder(data2, emb1)
    # print(f'CrossEncoderLayer: {data2.shape} {emb2.shape}')

    # CrossEncoderLayer init & call
    global_encoder = GlobalEncoderLayer(d_model=128, num_heads=2, dff=256, dropout_rate=0.1)
    emb1, emb2, emb3 = global_encoder(data1, data2, data3)
    print(f'GlobalEncoderLayer Temporal:  {data1.shape} {emb1.shape}')
    print(f'GlobalEncoderLayer Spatial:   {data2.shape} {emb2.shape}')
    print(f'GlobalEncoderLayer Exogenous: {data3.shape} {emb3.shape}')

    print('Hello World!')
