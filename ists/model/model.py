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

# if __name__ == '__main__':
#     # Input
#     data1 = np.random.randn(100000, 24, 4).astype(np.float32)
#     data2 = np.random.randn(100000, 12, 4).astype(np.float32)
#     data3 = np.random.randn(100000, 6, 4).astype(np.float32)
#     data = [data1, data2, data3, data3]
#     y = np.random.randn(100000, 1)
#
#     # Spatio-Temporal Transformer init and call
#     transformer = STTransformer(
#         feature_mask=[0, 0, 0, 0],
#         exg_feature_mask=[0, 0, 0, 0],
#         spatial_size=len(data) - 2,
#         kernel_size=3,
#         d_model=128,
#         num_heads=2,
#         dff=256,
#         fff=48,
#         dropout_rate=0.1,
#         null_max_size=None,
#         time_max_sizes=None,
#         exg_time_max_sizes=None,
#     )
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # , clipnorm=1.0, clipvalue=0.5)
#     transformer.compile(
#         loss='mse',
#         optimizer=optimizer,
#         metrics=['mae', 'mse']
#     )
#
#     transformer.fit(
#         x=data,
#         y=y,
#         epochs=3,
#         batch_size=32,
#         validation_split=0.2,
#         verbose=2
#     )
#
#     y_preds = transformer.predict(data)
#     print(y_preds.shape)
#     print('Hello World!')
