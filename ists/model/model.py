import numpy as np
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

    def __init__(self, base_model, hidden_units, activation='gelu'):
        super().__init__()
        if base_model == 'lstm':
            self.base = tf.keras.layers.LSTM(hidden_units)
        elif base_model == 'bilstm':
            self.base = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units))
        elif base_model == 'dense':
            self.base = tf.keras.layers.Dense(hidden_units, activation=activation)
        else:
            raise ValueError(f'Unsupported model {base_model}')

        self.dense = tf.keras.layers.Dense(hidden_units, activation=activation)
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, **kwargs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x
