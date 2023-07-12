from typing import List

import tensorflow as tf

from ists.model.embedding import TemporalEmbedding, SpatialEmbedding
from ists.model.encoder import GlobalEncoderLayer, EncoderLayer


class TTransformer(tf.keras.Model):
    def __init__(
            self,
            *,
            feature_mask,
            kernel_size,
            d_model,
            num_heads,
            dff,
            fff,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, x, **kwargs):
        x = self.temporal_embedder(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dense(x)
        pred = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return pred


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
            num_layers=1,
            with_cross=True,
            dropout_rate=0.1,
            null_max_size=None,
            time_max_sizes=None,
            exg_time_max_sizes=None,
    ):
        super().__init__()

        self.temporal_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=feature_mask,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )
        self.exogenous_embedder = TemporalEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            feature_mask=exg_feature_mask,
            time_max_sizes=exg_time_max_sizes,
        )
        self.spatial_embedder = SpatialEmbedding(
            d_model=d_model,
            kernel_size=kernel_size,
            spatial_size=spatial_size,
            feature_mask=feature_mask,
            null_max_size=null_max_size,
            time_max_sizes=time_max_sizes,
        )

        self.encoder = GlobalEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            num_layers=num_layers,
            with_cross=with_cross,
            dropout_rate=dropout_rate
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(fff, activation='gelu')
        self.final_layer = tf.keras.layers.Dense(1)
        self.last_attn_scores = None

    def call(self, inputs, **kwargs):
        temporal_x = inputs[0]
        exogenous_x = inputs[1]
        spatial_array = inputs[2:]

        temporal_x = self.temporal_embedder(temporal_x)
        exogenous_x = self.exogenous_embedder(exogenous_x)
        spatial_x = self.spatial_embedder(spatial_array)

        temporal_emb, exogenous_emb, spatial_emb = self.encoder(temporal_x, exogenous_x, spatial_x)
        embedded_x = tf.concat([temporal_emb, exogenous_emb, spatial_emb], axis=1)
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
