import numpy as np
import tensorflow as tf

import ists_.model.model
from ists_.model.embedding import PositionalEmbedding, TemporalEmbedding, create_fixed_variable_embeddings, \
    create_fixed_dense_variable_embeddings
from ists_.model.encoder import EncoderLayer
from ists_.model.pooling import AttentivePooling


# from ists_.model.rev_in import RevIN


class PositionalEmbeddingCLS(PositionalEmbedding):

    def call(self, x):
        return self.pe[:, 1:1 + tf.shape(x)[-2], :]


class TemporalEmbeddingCLS(TemporalEmbedding):

    def __init__(self, d_model, kernel_size, feature_mask, time_features=None, activation="relu", l2_reg=None):
        super().__init__(d_model, kernel_size, feature_mask, False, time_features, activation, l2_reg)
        self.pos_embedder = PositionalEmbeddingCLS(self.d_model)


def compute_masked_mean(x_raw, mask):
    masked_sum = tf.reduce_sum(x_raw * mask, axis=-1)
    masked_count = tf.reduce_sum(mask, axis=-1)
    mean = masked_sum / masked_count
    return mean


class IstfCLS(tf.keras.Model):

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
            encoder_layer_cls=None,
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
        self.encoder_layer_cls = encoder_layer_cls
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

        self.cls_token = self.add_weight(shape=(1, 1, 1, self.d_model), initializer="random_normal", trainable=True)

        encoder_layer_cls = {
            "MVEncoderLayerLA": EncoderLayer,
            "MVEncoderLayerGA": EncoderLayer,
        }.get(self.encoder_layer_cls, MVEncoderLayerLGA)
        shared_weights = True  # todo: add as argument
        # shared_weights = False  # todo: add as argument
        new_encoder_layer = lambda: encoder_layer_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            # shared_weights=shared_weights
        )
        if shared_weights:
            self.encoders = [new_encoder_layer()] * self.num_layers
        else:
            self.encoders = [new_encoder_layer() for _ in range(self.num_layers)]

        """self.predictor = Regressor(1, self.d_model*2, self.activation, self.dropout_rate, self.l2_reg)
        # self.auxiliary_head = Regressor(1, self.d_model*2, self.activation, self.dropout_rate, self.l2_reg)"""
        """self.seq = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.d_model*2, activation=self.activation, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
            tf.keras.layers.Dropout(self.dropout_rate),
        ])"""
        self.predictor = Regressor(1, 0, self.activation, self.dropout_rate, 0)
        # self.auxiliary_head = Regressor(1, 0, self.activation, self.dropout_rate, 0)
        # self.auxiliary_weight = 1.0

        self.static_feats_start = None
        if "static_feats_start" in kwargs:
            self.static_feats_start = kwargs["static_feats_start"]
            # self.static_embedder = Regressor(self.d_model, self.d_model*2, self.activation, self.dropout_rate, self.l2_reg)
            self.static_embedder = Regressor(self.d_model, 0, self.activation, self.dropout_rate, 0)

        self.variable_embeddings = None
        # self.rev_in : RevIN = None
        # self.pooling = AttentivePooling()

    def build(self, input_shape):
        V = input_shape[0][1]
        # fixed embeddings
        variable_embeddings = create_fixed_dense_variable_embeddings(V, self.d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]
        self.variable_embeddings = self.add_weight(
            shape=(1, V, 1, self.d_model),
            initializer=tf.keras.initializers.Constant(variable_embeddings),
            trainable=True,
            # trainable=False,
        )
        self.ve_scale = self.add_weight(
            name='ve_scale',
            shape=(),
            initializer='ones',
            trainable=True,
        )
        self.ve_bias = 0.0
        """self.ve_bias = self.add_weight(
            name='ve_bias',
            shape=(),
            initializer='zeros',
            trainable=True,
        )"""
        """# learnable embeddings
        self.variable_embeddings = tf.keras.layers.Embedding(
            input_dim=V,
            output_dim=self.d_model,
            embeddings_initializer="random_normal",
            trainable=True,
        )"""
        """# project smaller embeddings
        self.proj = tf.keras.layers.Dense(self.d_model, use_bias=False)
        variable_embeddings = create_fixed_variable_embeddings(V, V)  # (V, V)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (V, e) -> (1, V, 1, V)
        self.variable_embeddings = self.add_weight(
            shape=(1, V, 1, V),
            initializer=tf.keras.initializers.Constant(variable_embeddings),
            trainable=False,
        )"""
        # self.rev_in = RevIN(V, affine=False)

    def call(self, inputs):  # (b, v, t, f)
        exg_x, _ = inputs

        X = exg_x
        if self.static_feats_start is None:
            X_static = tf.zeros((1, 1), dtype=tf.float32)  # (1, 1)
        else:
            X_static = X[:, self.static_feats_start:, 0, self.raw_feature_id, tf.newaxis]  # (b, v_static, 1)
            V_static = tf.shape(X_static)[1]
            X_static = tf.reshape(X_static, (-1, 1))  # (b*v_static, 1)
            X_static = self.static_embedder(X_static)  # (b*v_static, e)
            X_static = tf.reshape(X_static, (-1, V_static, self.d_model))  # (b, v_static, e)
            X_static = tf.reduce_sum(X_static, axis=1)  # (b, e)
            X = X[:, :self.static_feats_start, :, :]
        X_shape = tf.shape(X)
        B, V, T = X_shape[0], X_shape[1], X_shape[2]

        if self.attn_mask_id is None:
            X, attn_mask = X, tf.ones((B, V, T), dtype=tf.float32)
        else:
            X, attn_mask = tf.gather(X, self.value_ids, axis=-1), tf.gather(X, self.attn_mask_id, axis=-1)

        # X_raw = X[:, :, :, self.raw_feature_id]  # (b, v, t, f) -> (b, v, t)
        # aux_true = compute_masked_mean(X_raw, attn_mask)  # (b, v)

        """X_raw = tf.transpose(X_raw, perm=[0, 2, 1])  # (b, v, t) -> (b, t, v)
        X_raw, X_stats = self.rev_in(X_raw, mode="norm", mask=tf.transpose(attn_mask, perm=[0, 2, 1]))  # (b, t, v)
        X_raw = tf.transpose(X_raw, perm=[0, 2, 1])  # (b, t, v) -> (b, v, t)
        X_raw = X_raw[:, :, :, tf.newaxis]  # (b, v, t) -> (b, v, t, 1)
        X = tf.concat([X[:, :, :, :self.raw_feature_id], X_raw, X[:, :, :, self.raw_feature_id + 1:]], axis=-1)"""

        X = tf.reshape(X, (V * B, T, -1))  # (b, v, t, f) -> (v*b, t, f)
        X = self.embedder(X)
        X = tf.reshape(X, (B, V, T, -1))  # (v*b, t, e) -> (b, v, t, e)

        # shared cls token
        cls_token = tf.tile(self.cls_token, [B, V, 1, 1])  # (1, 1, 1, e) -> (b, v, 1, e)
        cls_pe = self.embedder.pos_embedder.pe[:, 0:1, :][tf.newaxis]  # (1, 1, 1, e)
        cls_token = cls_token + cls_pe
        X_static = X_static[:, tf.newaxis, tf.newaxis, :]  # (b, e) -> (b, 1, 1, e)
        cls_token = cls_token + X_static
        X = tf.concat([cls_token, X], axis=2)  # (b, v, 1, e) + (b, v, t, e) -> (b, v, t+1, e)
        # X = X + X_static

        # fixed variable embeddings
        variable_embeddings = self.variable_embeddings
        variable_embeddings = variable_embeddings * self.ve_scale + self.ve_bias  # (1, V, 1, e) -> (1, V, 1, e)
        """# learnable variable embeddings
        variable_embeddings = self.variable_embeddings(tf.range(V))  # (v, e)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (v, e) -> (1, V, 1, e)"""
        """# project smaller embeddings
        variable_embeddings = self.proj(self.variable_embeddings)"""
        X = X + variable_embeddings

        attn_mask = tf.concat([tf.ones((B, V, 1)), attn_mask], axis=-1)  # (b, v, t) -> (b, v, t+1)

        X, attn_mask = tf.transpose(X, perm=[1, 0, 2, 3]), tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t+1, e) -> (v, b, t+1, e)
        X = self.dropout(X)

        for i in range(self.num_layers):
            # X = self.encoders[i](X, attention_mask=attn_mask)
            # mask only on first layers
            if i == self.num_layers - 1:
                X = self.encoders[i](X, do_global=False)
            else:
                X = self.encoders[i](X, attention_mask=attn_mask)

        X = self.dropout_1(X)
        cls_output = X[:, :, 0]  # cls_output: (v, b, e)
        cls_output = tf.transpose(cls_output, perm=[1, 0, 2])  # cls_output: (b, v, e)

        # cls_output = self.dropout_1(cls_output)

        # cls_output = self.seq(cls_output)  # cls_output: (b, v, e*2)

        """X = X[:, :, 1:]
        X = tf.reshape(X, (V*B, -1, self.d_model))  # (v, b, t, e) -> (v*b, t, e)
        pool_output = self.pooling(X)
        pool_output = tf.reshape(pool_output, (V, B, self.d_model))  # (v*b, e) -> (v, b, e)
        pool_output = tf.transpose(pool_output, perm=[1, 0, 2])  # (b, v, e)"""

        # cls_output = pool_output
        # cls_output = tf.concat([cls_output, pool_output], axis=-1)  # (b, v, e*2)

        # cls_output_1 = cls_output[:, 0]  # (b, e)
        """X = X[0, :, 1:]
        X_avg = tf.reduce_mean(X, axis=1)  # (b, t, e) -> (b, e)
        cls_output_1 = tf.concat([cls_output_1, X_avg], axis=1)  # (b, e*2)"""
        # pred = self.predictor(cls_output_1)

        pred = self.predictor(cls_output)  # (b, v, e) -> (b, v, 1)
        """pred = tf.transpose(pred, perm=[0, 2, 1])  # (b, 1, v)
        pred = self.rev_in(pred, mode="denorm", stats=X_stats)
        pred = tf.transpose(pred, perm=[0, 2, 1])  # (b, 1, v) -> (b, v, 1)"""
        pred = pred[:, 0]  # first variable only, (b, 1)

        '''# Auxiliary: predict mean of real (non-imputed) values
        cls_output_2 = cls_output
        aux_pred = tf.squeeze(self.auxiliary_head(cls_output_2), axis=-1)
        aux_loss = tf.reduce_mean(tf.square(aux_pred - aux_true))
        self.add_loss(self.auxiliary_weight * aux_loss)'''

        return pred


class MVEncoderLayerLGA(ists_.model.model.EncoderLocalGlobalAttnMaskLayer):

    def call(self, x, attention_mask=None, do_local=True, do_global=True):  # x: (v, b, t, e) attn_mask: (v, b, t)
        shape = tf.shape(x)
        v, b, t, e = shape[0], shape[1], shape[2], shape[3]

        attn_mask = attention_mask
        if attn_mask is None:
            attn_mask = tf.ones((v, b, t), dtype=tf.float32)

        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (b, v, t, e)
        attn_mask = tf.transpose(attn_mask, perm=[1, 0, 2])  # attn_mask: (b, v, t)

        if do_local:
            # attn_mask_loc = tf.reshape(attn_mask, (b*v, t, 1))  # attn_mask: (b*v, t, 1)  QMask
            attn_mask_loc = tf.reshape(attn_mask, (b*v, 1, t))  # attn_mask: (b*v, 1, t)  KMask
            """attn_mask_loc = tf.reshape(attn_mask, (b*v, t))  # attn_mask: (b*v, t)
            attn_mask_loc = tf.expand_dims(attn_mask_loc, -1) * tf.expand_dims(attn_mask_loc, 1)  # attn_mask: (b*v, t, t)  symmetric mask"""

            """# KMask but CLS attends every token
            attn_mask_loc = tf.reshape(attn_mask, (b * v, 1, t))  # attn_mask: (b*v, 1, t)
            attn_mask_loc = tf.tile(attn_mask_loc, [1, t, 1])  # attn_mask: (b*v, t, t)
            attn_mask_loc = tf.concat([tf.ones((b*v, 1, t)), attn_mask_loc[:, 1:]], axis=1)  # attn_mask: (b*v, t, t)"""

            x = tf.reshape(x, (b*v, t, e))  # x: (b*v, t, e)
            x = self.loc_attn(x, attention_mask=attn_mask_loc)
            x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        if do_global:
            # attn_mask_glb = tf.reshape(attn_mask, (b, v*t, 1))  # attn_mask: (b, v*t, 1)  QMask
            attn_mask_glb = tf.reshape(attn_mask, (b, 1, v*t))  # attn_mask: (b, 1, v*t)  KMask
            """attn_mask_glb = tf.reshape(attn_mask, (b, v*t))  # attn_mask: (b, v*t)
            attn_mask_glb = tf.expand_dims(attn_mask_glb, -1) * tf.expand_dims(attn_mask_glb, 1)  # attn_mask: (b, v*t, v*t)  symmetric mask"""

            """# KMask but CLS does not attend any token
            attn_mask_glb = tf.concat([tf.zeros((b, v, 1)), attn_mask[:, :, 1:]], axis=-1)
            mul = tf.ones((b, v, t))
            mul = tf.concat([tf.zeros((b, v, 1)), mul[:, :, 1:]], axis=-1)
            attn_mask_glb = tf.reshape(attn_mask_glb, (b, 1, v * t))  # attn_mask: (b, 1, v*t)
            mul = tf.reshape(mul, (b, v * t, 1))  # mul: (b, v*t, 1)
            attn_mask_glb = mul * attn_mask_glb  # attn_mask_glb: (b, v*t, v*t)"""

            x = tf.reshape(x, (b, v*t, e))  # x: (b, v*t, e)
            x = self.glb_attn(x, attention_mask=attn_mask_glb)
            x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        x = self.ffn(x)

        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (v, b, t, e)
        return x


def Regressor(output_size, hidden_units=0, activation="relu", dropout_rate=0.1, l2_reg=None):
    l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    seq = []
    if hidden_units > 0:
        seq.extend([
            tf.keras.layers.Dense(hidden_units, activation=activation, kernel_regularizer=l2_reg),
            tf.keras.layers.Dropout(dropout_rate)
        ])
    seq.append(tf.keras.layers.Dense(output_size, activation='linear'))
    return tf.keras.models.Sequential(seq)
