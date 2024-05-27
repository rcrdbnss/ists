from typing import List

import tensorflow as tf

from .embedding import TemporalEmbedding, SpatialEmbedding2
from .encoder import GlobalEncoderLayer


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
            do_exg=True, do_spt=True, do_glb=True, do_target=True
    ):
        super().__init__()
        self.do_exg, self.do_spt = do_exg, do_spt
        self.do_target = do_target

        self.exogenous_embedder = SpatialEmbedding2(
            TemporalEmbedding(
                d_model=d_model,
                kernel_size=kernel_size,
                feature_mask=feature_mask,
                with_cnn=exg_cnn,
                null_max_size=null_max_size,
                time_max_sizes=exg_time_max_sizes,
            )
        ) if self.do_exg else lambda x: None

        self.spatial_embedder = SpatialEmbedding2(
            TemporalEmbedding(
                d_model=d_model,
                kernel_size=kernel_size,
                feature_mask=feature_mask,
                with_cnn=spt_cnn,
                null_max_size=null_max_size,
                time_max_sizes=time_max_sizes,
            )
        ) if self.do_spt else lambda x: None

        self.encoder = GlobalEncoderLayer(
            d_model=d_model,
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
        exg_arr, spt_arr = inputs
        if (self.do_exg, self.do_spt, self.do_target) == (False, True, False):
            exg_arr = []
            spt_arr = spt_arr[1:]
        elif (self.do_exg, self.do_spt, self.do_target) == (False, True, True):
            exg_arr = []
        elif (self.do_exg, self.do_spt, self.do_target) == (True, False, False):
            exg_arr = exg_arr[1:]
            spt_arr = []
        elif (self.do_exg, self.do_spt, self.do_target) == (True, False, True):
            spt_arr = []
        elif (self.do_exg, self.do_spt, self.do_target) == (True, True, False):
            exg_arr = exg_arr[1:]
            spt_arr = spt_arr[1:]

        exg_x = self.exogenous_embedder(exg_arr)
        spt_x = self.spatial_embedder(spt_arr)
        exogenous_emb, spatial_emb = self.encoder((exg_x, spt_x))
        embedded_x = tf.concat([exogenous_emb, spatial_emb], axis=1)

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
