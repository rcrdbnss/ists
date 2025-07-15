import numpy as np
import tensorflow as tf

import ists_.model.model
from ists_.model.embedding import PositionalEmbedding, TemporalEmbedding, create_fixed_dense_variable_embeddings
from ists_.model.encoder import EncoderLayer
from ists_.model.model_cls import MVEncoderLayerLGA, compute_masked_mean, TemporalEmbeddingCLS
from ists_.model.pooling import AttentivePooling


class IstfAvgPool(tf.keras.Model):

    def __init__(
            self, *,
            feature_mask,
            kernel_size,
            d_model,
            num_heads,
            dff,
            activation='relu',
            num_layers=1,
            dropout_rate=0.1,
            time_features=None,
            do_exg=True, do_spt=True, do_emb=True, force_target=False,
            l2_reg=None,
            **kwargs
    ):
        super().__init__()

        feature_mask = np.array(feature_mask)
        value_ids = np.arange(len(feature_mask))
        arg_null = feature_mask == 1
        if arg_null.any():
            self.feature_mask, self.value_ids = feature_mask[~arg_null], value_ids[~arg_null]
            self.attn_mask_id = value_ids[arg_null][0]
        else:
            self.feature_mask, self.value_ids = feature_mask, value_ids
            self.attn_mask_id = None
        self.raw_feature_id = np.where(self.feature_mask == 0)[0][0]

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.activation = activation
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.time_features = time_features
        self.do_exg, self.do_spt, self.do_emb, self.force_target = do_exg, do_spt, do_emb, force_target
        self.l2_reg = l2_reg

        self.embedder = TemporalEmbeddingCLS(
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            feature_mask=self.feature_mask,
            time_features=self.time_features,
            activation=self.activation,
            l2_reg=self.l2_reg
        )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)

        encoder_layer_cls = MVEncoderLayerLGA
        shared_weights = True  # todo: add as argument
        # shared_weights = False  # todo: add as argument
        new_encoder_layer = lambda: encoder_layer_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )
        if shared_weights:
            self.encoders = [new_encoder_layer()] * self.num_layers
        else:
            self.encoders = [new_encoder_layer() for _ in range(self.num_layers)]

        self.predictor = tf.keras.layers.Dense(1, activation='linear')
        # self.aux_mean_head = tf.keras.layers.Dense(1, activation='linear')
        # self.aux_intp_head = tf.keras.layers.Dense(1, activation='linear')
        # self.aux_mean_weight, self.aux_intp_weight = 1.0, 0.1

        self.variable_embeddings = None
        self.pooling = AttentivePooling()

    def build(self, input_shape):
        V = input_shape[0][1]
        variable_embeddings = create_fixed_dense_variable_embeddings(V, self.d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]
        self.variable_embeddings = self.add_weight(
            shape=(1, V, 1, self.d_model),
            initializer=tf.keras.initializers.Constant(variable_embeddings),
            trainable=True,
            # trainable=False,
        )

    def call(self, inputs):  # (b, v, t, f)
        exg_x, _ = inputs

        X = exg_x
        X_shape = tf.shape(X)
        B, V, T = X_shape[0], X_shape[1], X_shape[2]

        if self.attn_mask_id is None:
            X, attn_mask = X, tf.ones((B, V, T), dtype=tf.float32)
        else:
            X, attn_mask = tf.gather(X, self.value_ids, axis=-1), tf.gather(X, self.attn_mask_id, axis=-1)

        # X_raw = X[:, :, :, self.raw_feature_id]  # (b, v, t, f) -> (b, v, t)
        # aux_mean_true = compute_masked_mean(X_raw, attn_mask)  # (b, v, t)
        # aux_intp_true = tf.reshape(X_raw, (V, B, T))  # (b, v, t) -> (v, b, t)

        X = tf.reshape(X, (V * B, T, -1))  # (b, v, t, f) -> (v*b, t, f)
        X = self.embedder(X)
        X = tf.reshape(X, (B, V, T, self.d_model))  # (v*b, t, e) -> (b, v, t, e)

        X = X + self.variable_embeddings  # (b, v, t, e) + (1, v, 1, e) -> (b, v, t, e)

        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (b, v, t, e) -> (v, b, t, e)
        attn_mask = tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t) -> (v, b, t)

        X_enc = X
        X_enc = self.dropout(X_enc)
        for i in range(self.num_layers):
            X_enc = self.encoders[i](X_enc, attention_mask=attn_mask)

        X_enc = self.dropout_1(X_enc)
        # X_pool = tf.reduce_mean(X_enc[0], axis=1)  # (b, t, e) -> (b, e)
        X_enc = tf.reshape(X_enc, (V*B, -1, self.d_model))  # (v, b, t, e) -> (v*b, t, e)
        X_pool = self.pooling(X_enc)
        X_pool = tf.reshape(X_pool, (V, B, self.d_model))  # (v*b, t, e) -> (v, b, e)
        X_pool = tf.transpose(X_pool, perm=[1, 0, 2])  # (v, b, e) -> (b, v, e)
        pred = self.predictor(X_pool)  # (b, v, e) -> (b, v, 1)
        pred = pred[:, 0]  # first variable only

        return pred
