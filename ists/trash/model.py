import tensorflow as tf

from ists.model.embedding import TemporalEmbedding, SpatialEmbedding2, SpatialEmbeddingAsMultiVariate
from .encoder import ContextualEncoder, GlobalEncoderLayer


class STTransformer(tf.keras.Model):
    def __init__(
            self,
            *,
            feature_mask,
            # exg_feature_mask,
            spatial_size,
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
            with_cross=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            # exg_time_max_sizes=None,
            do_exg=True, do_spt=True, do_glb=True, do_emb=True, force_target=False,
            exg_size=None,
            multivar=False,
            encoder_cls=GlobalEncoderLayer,
            encoder_layer_cls=None,
            l2_reg=None
    ):
        super().__init__()
        self.do_exg, self.do_spt = do_exg, do_spt
        self.force_target = force_target

        # if univar_or_multivar == 'univar':
        if multivar:
            ExogenousEmbedding = spatial_embedding_as_multivariate(feature_mask, exg_size)
            SpatialEmbedding = spatial_embedding_as_multivariate(feature_mask, spatial_size)
        # elif univar_or_multivar == 'multivar':
        else:
            ExogenousEmbedding = spatial_embedding(feature_mask)
            SpatialEmbedding = spatial_embedding(feature_mask)

        self.exogenous_embedder = ExogenousEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            with_cnn=exg_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        ) if self.do_exg or force_target else lambda x: tf.zeros((tf.shape(x[0])[0], 0, d_model))  # lambda x: None
        if not do_emb:
            del self.exogenous_embedder
            self.exogenous_embedder = lambda x: tf.concat(x, axis=1)

        self.spatial_embedder = SpatialEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            with_cnn=spt_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        ) if self.do_spt or force_target else lambda x: tf.zeros((tf.shape(x[0])[0], 0, d_model))  # lambda x: None
        if not do_emb:
            del self.spatial_embedder
            self.spatial_embedder = lambda x: tf.concat(x, axis=1)

        if force_target and not do_exg and not do_spt:
            # leave only spatial branch on
            del self.exogenous_embedder
            self.exogenous_embedder = lambda x: tf.zeros((tf.shape(x[0])[0], 0, d_model))  # None

        self.encoder = encoder_cls(
            d_model=(d_model if do_emb else len(feature_mask)),
            num_heads=num_heads,
            dff=dff,
            activation=activation,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            do_exg=do_exg, do_spt=do_spt, do_glb=do_glb, # todo: force_target
            layer_cls=encoder_layer_cls,
            l2_reg=l2_reg
        )

        self.flatten = tf.keras.layers.Flatten()
        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)
        self.dense = tf.keras.layers.Dense(fff, activation='gelu', **reg)
        self.final_layer = tf.keras.layers.Dense(1, **reg)
        self.last_attn_scores = None

    def call(self, inputs, **kwargs):
        exg_x, spt_x = inputs
        if self.force_target and self.do_spt and not self.do_exg:
            exg_x = [exg_x[0]]
        if self.force_target and not self.do_spt:
            spt_x = [spt_x[0]]

        exg_x = self.exogenous_embedder(exg_x)
        spt_x = self.spatial_embedder(spt_x)
        exg_x = tf.concat(exg_x, axis=1)
        spt_x = tf.concat(spt_x, axis=1)
        exg_x, spt_x = self.encoder((exg_x, spt_x))
        embedded_x = tf.concat([exg_x, spt_x], axis=1)

        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred


class STTransformerContextualEnc(STTransformer):

    def __init__(
            self,
            *,
            feature_mask,
            # exg_feature_mask,
            spatial_size,
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
            with_cross=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            exg_time_max_sizes=None,
            do_exg=True, do_spt=True, do_glb=True, do_emb=True, force_target=False,
            exg_size=None,
            multivar=False,
            encoder_layer_cls=None,
            l2_reg=None
    ):
        super().__init__(feature_mask=feature_mask, spatial_size=spatial_size,  # exg_feature_mask=exg_feature_mask,
                         kernel_size=kernel_size, d_model=d_model, num_heads=num_heads, dff=dff, fff=fff,
                         activation=activation, exg_cnn=exg_cnn, spt_cnn=spt_cnn, time_cnn=time_cnn,
                         num_layers=num_layers,
                         with_cross=with_cross, dropout_rate=dropout_rate, null_max_size=null_max_size,
                         time_max_sizes=time_max_sizes, do_exg=do_exg,  # exg_time_max_sizes=exg_time_max_sizes,
                         do_spt=do_spt, do_glb=do_glb, do_emb=do_emb, force_target=force_target, exg_size=exg_size,
                         multivar=multivar, encoder_cls=ContextualEncoder, encoder_layer_cls=encoder_layer_cls,
                         l2_reg=l2_reg)

        self.target_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            with_cnn=time_cnn,
            time_max_sizes=time_max_sizes,
            null_max_size=null_max_size
        )

    def call(self, inputs, **kwargs):
        exg_x, spt_x = inputs
        if self.force_target and self.do_spt and not self.do_exg:
            exg_x = [exg_x[0]]
        if self.force_target and not self.do_spt:
            spt_x = [spt_x[0]]

        # assuming that each element has the same shape
        num_past = min(exg_x[0].shape[1], spt_x[0].shape[1])
        x = spt_x[0][:, -num_past:]
        exg_x = tuple([x[:, -num_past:] for x in exg_x][1:])
        spt_x = tuple([x[:, -num_past:] for x in spt_x][1:])

        x = self.target_embedder(x)
        exg_x = self.exogenous_embedder(exg_x)
        spt_x = self.spatial_embedder(spt_x)
        b, t, e = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        x, exg_x, spt_x = self.encoder((x, exg_x, spt_x))

        x = tf.reshape(x, (-1, t, e))
        x = self.flatten(x)
        x = self.dense(x)
        pred = self.final_layer(x)

        return pred


def spatial_embedding(feature_mask):
    def init(d_model, kernel_size, with_cnn=True, time_max_sizes=None, null_max_size=None):
        return SpatialEmbedding2(TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            with_cnn=with_cnn,
            time_max_sizes=time_max_sizes,
            null_max_size=null_max_size
        ))

    return init


def spatial_embedding_as_multivariate(feature_mask, n_univars):
    def init(d_model, kernel_size, with_cnn=True, time_max_sizes=None, null_max_size=None):
        return SpatialEmbeddingAsMultiVariate(
            d_model=d_model,
            kernel_size=kernel_size,
            n_univars=n_univars,
            feature_mask_univar=feature_mask,
            with_cnn=with_cnn,
            time_max_sizes=time_max_sizes,
            null_max_size=null_max_size
        )

    return init
