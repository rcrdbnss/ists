from typing import List, Literal

import tensorflow as tf

from .embedding import TemporalEmbedding, SpatialEmbedding2, SpatialEmbeddingAsMultiVariate
from .encoder import GlobalEncoderLayer, SpatialExogenousEncoder


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


class STTransformer(tf.keras.Model):
    def __init__(
            self,
            *,
            feature_mask,
            exg_feature_mask,
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
            multivar = False,
            encoder_cls=GlobalEncoderLayer
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
            time_max_sizes=exg_time_max_sizes,
        ) if self.do_exg or force_target else lambda x: tf.zeros((tf.shape(x[0])[0], 0, d_model)) # lambda x: None
        if not do_emb:
            del self.exogenous_embedder
            self.exogenous_embedder = lambda x: tf.concat(x, axis=1)

        self.spatial_embedder = SpatialEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            with_cnn=spt_cnn,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        ) if self.do_spt or force_target else lambda x: tf.zeros((tf.shape(x[0])[0], 0, d_model)) # lambda x: None
        if not do_emb:
            del self.spatial_embedder
            self.spatial_embedder = lambda x: tf.concat(x, axis=1)

        if force_target and not do_exg and not do_spt:
            # leave only spatial branch on
            del self.exogenous_embedder
            self.exogenous_embedder = lambda x: tf.zeros((tf.shape(x[0])[0], 0, d_model)) # None

        # # todo: rewrite better
        # if encoder_cls == SpatialExogenousEncoder:
        #     del self.exogenous_embedder
        #     del self.spatial_embedder
        #     self.__exg_emb = TemporalEmbedding(
        #         d_model=d_model,
        #         kernel_size=kernel_size,
        #         feature_mask=feature_mask,
        #         with_cnn=exg_cnn,
        #         time_max_sizes=time_max_sizes,
        #         null_max_size=null_max_size
        #     )
        #     self.__spt_emb = TemporalEmbedding(
        #         d_model=d_model,
        #         kernel_size=kernel_size,
        #         feature_mask=feature_mask,
        #         with_cnn=spt_cnn,
        #         time_max_sizes=time_max_sizes,
        #         null_max_size=null_max_size
        #     )
        #     self.exogenous_embedder = lambda inputs: [self.__exg_emb(x) for x in inputs]
        #     self.spatial_embedder = lambda inputs: [self.__spt_emb(x) for x in inputs]

        self.encoder = encoder_cls(
            d_model=(d_model if do_emb else len(feature_mask)),
            num_heads=num_heads,
            dff=dff,
            activation=activation,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            do_exg=do_exg, do_spt=do_spt, do_glb=do_glb,
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)
        self.last_attn_scores = None

    def call(self, inputs, **kwargs):
        exg_x, spt_x = inputs
        if self.force_target and self.do_spt and not self.do_exg:
            exg_x = [exg_x[0]]
        if self.force_target and not self.do_spt:
            spt_x = [spt_x[0]]

        exg_x = self.exogenous_embedder(exg_x)
        spt_x = self.spatial_embedder(spt_x)
        exg_x, spt_x = self.encoder((exg_x, spt_x))
        embedded_x = tf.concat([exg_x, spt_x], axis=1)

        embedded_x = self.flatten(embedded_x)
        embedded_x = self.dense(embedded_x)
        pred = self.final_layer(embedded_x)

        return pred


class BaselineModel(tf.keras.Model):

    def __init__(
            self,
            feature_mask: List[int],
            base_model: str,
            hidden_units: int,
            skip_na: bool = False,
            activation: str = 'gelu'
    ):
        super().__init__()
        if base_model.startswith('lstm'):
            self.base = tf.keras.layers.LSTM(hidden_units)
        elif base_model.startswith('bilstm'):
            self.base = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units))
        elif base_model == 'dense':
            self.base = tf.keras.layers.Dense(hidden_units, activation=activation)
        else:
            raise ValueError(f'Unsupported model {base_model}')

        if skip_na and base_model == 'dense':
            raise ValueError('Impossible skip nan values with FeedForward Neural Network (dense)')

        if skip_na and 1 not in feature_mask:
            raise ValueError('Impossible skip nan values without nan encoding')

        self.dense = tf.keras.layers.Dense(hidden_units, activation=activation)
        self.final_layer = tf.keras.layers.Dense(1)

        self.feature_mask = feature_mask
        self.skip_na = skip_na
        self.feat_ids = [i for i, x in enumerate(feature_mask) if x == 0]
        self.null_id = [i for i, x in enumerate(feature_mask) if x == 1]

    def call(self, inputs, **kwargs):
        x = inputs[0]

        if x.shape[2] != len(self.feature_mask):
            raise ValueError('Input data have a different features dimension that the provided feature mask')

        # Extract value, null, and time array from the input matrix
        x_feat = tf.gather(x, self.feat_ids, axis=2)

        # Add the null encoding
        if self.skip_na:
            x_null = x[:, :, self.null_id[0]]
            mask = tf.equal(x_null, 0)
            x = self.base(x_feat, mask=mask)
        else:
            x = self.base(x_feat)

        x = self.dense(x)
        x = self.final_layer(x)
        return x
