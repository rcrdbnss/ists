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
    def __init__(self, d_model, dff, activation='relu', dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),
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
    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            activation=activation,
        )
        self.last_attn_scores = None

    def call(self, x, **kwargs):
        x = self.self_attention(x)
        self.last_attn_scores = self.self_attention.last_attn_scores
        x = self.ffn(x)

        return x


class CrossEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1):
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

        self.ffn = FeedForward(d_model, dff, activation=activation)
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

    def __init__(self, *, d_model, num_heads, dff, activation='relu', num_layers=1, dropout_rate=0.1,
                 do_exg=False, do_spt=True, do_glb=True,
                 ):
        super(GlobalEncoderLayer, self).__init__()

        self.num_layers = num_layers
        # self.do_exg = do_exg

        self.spatial_encoders = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(self.num_layers)
        ] if do_spt else [lambda x: x for _ in range(self.num_layers)]

        self.exogenous_encoders = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(self.num_layers)
        ] if do_exg else [lambda x: x for _ in range(self.num_layers)]

        self.global_encoders = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
                activation=activation,
            )
            for _ in range(self.num_layers)
        ] if do_glb else [lambda x: x for _ in range(self.num_layers)]

        # self.last_attn_scores = None

    def call(self, x, **kwargs):
        exg_x, spt_x = x

        if exg_x is None:
            _shape = spt_x.shape
            exg_x = np.random.random((_shape[0], 0, _shape[2]))
        if spt_x is None:
            _shape = exg_x.shape
            spt_x = np.random.random((_shape[0], 0, _shape[2]))

        for i in range(self.num_layers):
            exg_x = self.exogenous_encoders[i](exg_x)
            spt_x = self.spatial_encoders[i](spt_x)

            x = tf.concat([exg_x, spt_x], axis=1)
            x = self.global_encoders[i](x)
            exg_x, spt_x = tf.split(x, [exg_x.shape[1], spt_x.shape[1]], axis=1)

        return exg_x, spt_x


if __name__ == '__main__':
    # Input
    data1 = np.random.randn(64, 96, 64).astype(np.float32)
    data2 = np.random.randn(64, 480, 64).astype(np.float32)
    data3 = np.random.randn(64, 240, 64).astype(np.float32)

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
    emb2, emb3 = global_encoder((data2, data3))
    # print(f'GlobalEncoderLayer Temporal:  {data1.shape} {emb1.shape}')
    print(f'GlobalEncoderLayer Exogenous: {data2.shape} {emb2.shape}' if emb2 is not None else 'No Exogenous')
    print(f'GlobalEncoderLayer Spatial:   {data3.shape} {emb3.shape}')

    print('Hello World!')
