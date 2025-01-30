import numpy as np
import tensorflow as tf

from ists.model.embedding import TemporalEmbedding
from ists.model.encoder import GlobalSelfAttention, FeedForward, EncoderAttnMaskLayer


class STTransformerSequentialAttnMask(tf.keras.Model):

    def __init__(
            self,
            *,
            feature_mask,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            activation='relu',
            exg_cnn=True,
            spt_cnn=True,
            time_cnn=True,
            num_layers=1,
            dropout_rate=0.1,
            time_max_sizes=None,
            do_exg=True, do_spt=True, do_emb=True, force_target=False,
            encoder_layer_cls=None,
            l2_reg=None,
            **kwargs
    ):
        super().__init__()
        feature_mask = np.array(feature_mask)
        feature_ids = np.array([i for i, x in enumerate(feature_mask)])
        arg_null = feature_mask == 1
        if arg_null.any():
            self.feature_ids, self.feature_mask, self.null_id = feature_ids[~arg_null], feature_mask[~arg_null], feature_mask[arg_null][0]
        else:
            self.feature_ids, self.feature_mask, self.null_id = feature_ids, feature_mask, None
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.fff = fff
        self.activation = activation
        self.exg_cnn = exg_cnn
        self.spt_cnn = spt_cnn
        self.time_cnn = time_cnn
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.time_max_sizes = time_max_sizes
        self.do_exg, self.do_spt, self.do_emb, self.force_target = do_exg, do_spt, do_emb, force_target
        self.encoder_layer_cls = encoder_layer_cls
        self.l2_reg = l2_reg

        if do_emb:
            self.target_embedder = TemporalEmbedding(
                d_model=d_model,
                kernel_size=kernel_size,
                feature_mask=self.feature_mask,
                with_cnn=time_cnn,
                time_max_sizes=time_max_sizes,
            )
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        else:
            feature_mask_selector = np.isin(self.feature_mask, [0, 1])
            for i in range(len(feature_mask_selector)):
                if not feature_mask_selector[i]:
                    feature_mask_selector[i] = True
                    break
            self.feature_mask = self.feature_mask[feature_mask_selector]

            arg_time = (self.feature_mask == 2)
            if arg_time.any():
                def norm_time(x):
                    T = tf.shape(x)[1]
                    t = tf.linspace(0.0, 1.0, T)
                    t = tf.reshape(t, (1, -1, 1))
                    x = tf.boolean_mask(x, feature_mask_selector, axis=2)
                    x = tf.where(arg_time, t, x)
                    return x

                self.target_embedder = tf.keras.layers.Lambda(norm_time)
            else:
                self.target_embedder = tf.keras.layers.Lambda(lambda x: x)
            self.dropout = tf.keras.layers.Lambda(lambda x: x)

        # self.exogenous_embedder = SpatialEmbedding2(TemporalEmbedding(
        #     d_model=d_model,
        #     kernel_size=kernel_size,
        #     feature_mask=self.feature_mask,
        #     with_cnn=exg_cnn,
        #     time_max_sizes=time_max_sizes,
        # )) # if self.do_exg else lambda x: None # self.__dummy(d_model)
        # if not do_emb:
        #     del self.exogenous_embedder
        #     self.exogenous_embedder = lambda x: tf.concat(x, axis=1)
        #
        # self.spatial_embedder = SpatialEmbedding2(TemporalEmbedding(
        #     d_model=d_model,
        #     kernel_size=kernel_size,
        #     feature_mask=self.feature_mask,
        #     with_cnn=spt_cnn,
        #     time_max_sizes=time_max_sizes,
        # )) # if self.do_spt else lambda x: None # self.__dummy(d_model)
        # if not do_emb:
        #     del self.spatial_embedder
        #     self.spatial_embedder = lambda x: tf.concat(x, axis=1)

        self.encoder = SequentialEncoderAttnMask(
            d_model=(d_model if do_emb else len(self.feature_mask)),
            num_heads=num_heads,
            dff=dff,
            activation=activation,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            do_exg=do_exg, do_spt=do_spt, force_target=force_target,
            layer_cls=encoder_layer_cls,
            l2_reg=l2_reg,
        )

        # self.final_layers = FinalLayersFF(fff, dropout_rate, l2_reg)
        self.final_layers = FinalLayersGRU(fff, dropout_rate, l2_reg, gru_bidirectional=False)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'feature_mask': self.feature_mask.tolist(),
            'kernel_size': self.kernel_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'fff': self.fff,
            'activation': self.activation,
            'exg_cnn': self.exg_cnn,
            'spt_cnn': self.spt_cnn,
            'time_cnn': self.time_cnn,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'time_max_sizes': self.time_max_sizes,
            'do_exg': self.do_exg,
            'do_spt': self.do_spt,
            'do_emb': self.do_emb,
            'force_target': self.force_target,
            'encoder_layer_cls': self.encoder_layer_cls,
            'l2_reg': self.l2_reg
        })
        return config

    def split_data_attn_mask(self, x):
        if self.null_id is None:
            return x, None
        return tf.gather(x, self.feature_ids, axis=-1), 1 - tf.gather(x, self.null_id, axis=-1)

    def call(self, inputs):
        exg_x, spt_x = inputs[0], inputs[1]
        exg_x = tf.transpose(exg_x, (1, 0, 2, 3))
        spt_x = tf.transpose(spt_x, (1, 0, 2, 3))

        x = spt_x[0]
        exg_x = exg_x[1:]
        spt_x = spt_x[1:]

        x, x_attn_mask = self.split_data_attn_mask(x)
        exg_x, exg_attn_mask = list(zip(*[self.split_data_attn_mask(e) for e in tf.unstack(exg_x)]))
        # spt_x, spt_attn_mask = list(zip(*[self.split_data_attn_mask(s) for s in tf.unstack(spt_x)]))
        spt_x = tf.unstack(spt_x)
        if not self.do_spt:
            spt_x = []
        spt_x, spt_attn_mask = (
            ([], [None]) if (len(spt_x) == 0)
            else tuple(zip(*[self.split_data_attn_mask(s) for s in spt_x]))
        )
        if np.all([m is None for m in exg_attn_mask]): exg_attn_mask = None
        if np.all([m is None for m in spt_attn_mask]): spt_attn_mask = None

        # x = self.target_embedder(x)
        # exg_x = self.exogenous_embedder(exg_x)
        # spt_x = self.spatial_embedder(spt_x)
        x = self.target_embedder(x)
        exg_x = [self.target_embedder(e) for e in exg_x]
        spt_x = [self.target_embedder(s) for s in spt_x]
        x_shape = tf.shape(x)
        b, t, e = x_shape[0], x_shape[1], x_shape[2]
        x, exg_x, spt_x = self.dropout(x), [self.dropout(e) for e in exg_x], [self.dropout(s) for s in spt_x]
        x, exg_x, spt_x = self.encoder(x=x, exg_ctx=exg_x, spt_ctx=spt_x, x_attn_mask=x_attn_mask, exg_ctx_attn_mask=exg_attn_mask, spt_ctx_attn_mask=spt_attn_mask)

        x = tf.reshape(x, (b, t, e))
        pred = self.final_layers(x)  # CUDNN does not support masking

        return pred


class SequentialEncoderAttnMask(tf.keras.layers.Layer):

    def __switched_off(self, x, attn_mask=None):
        return x

    def __init__(self, *, d_model, num_heads, dff, activation='relu', num_layers=1, dropout_rate=0.1, l2_reg=None,
                 do_exg=True, do_spt=True, force_target=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.do_exg, self.do_spt = do_exg, do_spt
        if 'layer_cls' in kwargs and kwargs['layer_cls']:
            if kwargs['layer_cls'] == 'EncoderAttnMaskLayer':
                layer_cls = EncoderAttnMaskLayer
        else:
            layer_cls = EncoderLocalGlobalAttnMaskLayer

        if force_target and not do_exg and not do_spt:
            self.T = True
            self.spt_encs = [layer_cls(
                d_model=d_model, num_heads=num_heads, dff=dff, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg
            ) for _ in range(num_layers)]
        else:
            self.T = False

            self.exg_encs = [layer_cls(
                d_model=d_model, num_heads=num_heads, dff=dff, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg
            ) for _ in range(num_layers)] if do_exg else [self.__switched_off for _ in range(self.num_layers)]

            self.spt_encs = [layer_cls(
                d_model=d_model, num_heads=num_heads, dff=dff, activation=activation, dropout_rate=dropout_rate, l2_reg=l2_reg
            ) for _ in range(num_layers)] if do_spt else [self.__switched_off for _ in range(self.num_layers)]

    def call(self, x, exg_ctx, spt_ctx, x_attn_mask=None, exg_ctx_attn_mask=None, spt_ctx_attn_mask=None):
        x = tf.expand_dims(x, axis=0)
        if x_attn_mask is not None:
            x_attn_mask = tf.expand_dims(x_attn_mask, axis=0)
        if self.T:
            for i in range(self.num_layers):
                x = self.spt_encs[i](x=x, attn_mask=x_attn_mask)
        else:
            if x_attn_mask is None and exg_ctx_attn_mask is None:
                exg_attn_mask = None
            else: # elif x_attn_mask is not None and exg_ctx_attn_mask is not None:
                exg_attn_mask = tf.concat([x_attn_mask, exg_ctx_attn_mask], axis=0)
            if not self.do_spt:
                spt_ctx = tf.zeros_like(x)[0:0]
                if x_attn_mask is not None: spt_ctx_attn_mask = tf.zeros_like(x_attn_mask)[0:0]
            if x_attn_mask is None and spt_ctx_attn_mask is None:
                spt_attn_mask = None
            else: # elif x_attn_mask is not None and spt_ctx_attn_mask is not None:
                spt_attn_mask = tf.concat([x_attn_mask, spt_ctx_attn_mask], axis=0)
            for i in range(self.num_layers):
                exg_x = tf.concat([x, exg_ctx], axis=0)
                exg_x = self.exg_encs[i](exg_x, attn_mask=exg_attn_mask)
                x, exg_ctx = tf.split(exg_x, [1, tf.shape(exg_ctx)[0]], axis=0)
                spt_x = tf.concat([x, spt_ctx], axis=0)
                spt_x = self.spt_encs[i](spt_x, attn_mask=spt_attn_mask)
                x, spt_ctx = tf.split(spt_x, [1, tf.shape(spt_ctx)[0]], axis=0)
        # x = x[0]
        if not self.do_exg:
            exg_ctx = tf.zeros_like(exg_ctx)[0:0]
        if not self.do_spt:
            spt_ctx = tf.zeros_like(spt_ctx)[0:0]
        return x, exg_ctx, spt_ctx


class EncoderLocalGlobalAttnMaskLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None):
        super().__init__()

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.loc_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.glb_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate, **reg
        )

    def call(self, x, attn_mask=None):  # x: (v, b, t, e) attn_mask: (v, b, t)
        shape = tf.shape(x)
        v, b, t, e = shape[0], shape[1], shape[2], shape[3]

        if attn_mask is None:
            attn_mask_loc = None
        else:
            attn_mask_loc = tf.reshape(attn_mask, (v*b, t))  # attn_mask: (v*b, t)
            attn_mask_loc = tf.expand_dims(attn_mask_loc, axis=-1) * tf.expand_dims(attn_mask_loc, axis=1)  # attn_mask: (v*b, t, t)
        x = tf.reshape(x, (v*b, t, e))  # x: (v*b, t, e)
        x = self.loc_attn(x, attention_mask=attn_mask_loc)

        if attn_mask is None:
            attn_mask_glb = None
        else:
            attn_mask_glb = tf.transpose(attn_mask, perm=[1, 0, 2])  # attn_mask: (b, v, t)
            attn_mask_glb = tf.reshape(attn_mask_glb, (b, v*t))  # attn_mask: (b, v*t)
            attn_mask_glb = tf.expand_dims(attn_mask_glb, axis=-1) * tf.expand_dims(attn_mask_glb, axis=1)  # attn_mask: (b, v*t, v*t)
        x = tf.reshape(x, (v, b, t, e))  # x: (v, b, t, e)
        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (b, v, t, e)
        x = tf.reshape(x, (b, v*t, e))  # x: (b, v*t, e)
        x = self.glb_attn(x, attention_mask=attn_mask_glb)

        x = self.ffn(x)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)
        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (v, b, t, e)
        return x


def FinalLayersFF(fff, dropout_rate, l2_reg):
    l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(fff, activation='gelu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(fff, activation='gelu', kernel_regularizer=l2_reg),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='linear')
    ])


class FinalLayersGRU(tf.keras.layers.Layer):

    def __init__(self, fff, dropout_rate, l2_reg, *, gru_bidirectional=False):
        super().__init__()
        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.gru = tf.keras.layers.GRU(fff, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        if gru_bidirectional:
            self.gru = tf.keras.layers.Bidirectional(self.gru)
        self.final_layers = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(fff, activation='gelu', kernel_regularizer=l2_reg),
            tf.keras.layers.Dropout(dropout_rate),
            # tf.keras.layers.Dense(64, activation='gelu', kernel_regularizer=l2_reg),
            # tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def call(self, x, mask=None):
        x = self.gru(x, mask=mask)
        x = self.final_layers(x)
        return x
