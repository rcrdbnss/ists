import numpy as np
import tensorflow as tf

from ists_.model.model_cls import MVEncoderLayerLGA, compute_masked_mean, TemporalEmbeddingCLS


class IstfInterpCLS(tf.keras.Model):

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
            time_features=self.time_features,
            activation=self.activation,
            l2_reg=self.l2_reg
        )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.cls_token = self.add_weight(shape=(1, 1, 1, self.d_model), initializer="random_normal", trainable=True)

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

        """self.seq = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.d_model*2, activation=self.activation, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
            tf.keras.layers.Dropout(self.dropout_rate),
        ])"""
        self.predictor = tf.keras.layers.Dense(1, activation='linear')
        # self.aux_mean_head = tf.keras.layers.Dense(1, activation='linear')
        # self.aux_intp_head = tf.keras.layers.Dense(1, activation='linear')
        # self.aux_mean_weight, self.aux_intp_weight = 1.0, 0.1
        self.aux_recn_head = None

    def build(self, input_shape):
        T = input_shape[0][2]
        self.aux_recn_head = tf.keras.layers.Dense(T, activation='linear')
        # super(IstfInterpCLS, self).build(input_shape)

    def call(self, inputs):  # (b, v, t, f)
        exg_x, _ = inputs

        X = exg_x
        aux_mask, X = X[:, :, :, -1], X[:, :, :, :-1]
        X_shape = tf.shape(X)
        B, V, T = X_shape[0], X_shape[1], X_shape[2]

        if self.attn_mask_id is None:
            X, attn_mask = X, tf.ones((B, V, T), dtype=tf.float32)
        else:
            X, attn_mask = tf.gather(X, self.value_ids, axis=-1), tf.gather(X, self.attn_mask_id, axis=-1)

        X_raw = X[:, :, :, self.raw_feature_id]  # (b, v, t, f) -> (b, v, t)
        # aux_mean_true = compute_masked_mean(X_raw, attn_mask)  # (b, v, t)
        # aux_intp_true = tf.reshape(X_raw, (V, B, T))  # (b, v, t) -> (v, b, t)
        aux_recn_true = tf.reshape(X_raw, (V, B, T))  # (b, v, t) -> (v, b, t)

        X = tf.reshape(X, (V * B, T, -1))  # (b, v, t, f) -> (v*b, t, f)
        X = self.embedder(X)
        X = tf.reshape(X, (B, V, T, -1))  # (v*b, t, e) -> (b, v, t, e)

        # shared cls token
        cls_token = tf.tile(self.cls_token, [B, V, 1, 1])  # (1, 1, 1, e) -> (b, v, 1, e)
        cls_pe = self.embedder.pos_embedder.pe[:, 0:1, :][tf.newaxis]  # (1, 1, 1, e)
        cls_token = cls_token + cls_pe
        X = tf.concat([cls_token, X], axis=2)  # (b, v, 1, e) + (b, v, t, e) -> (b, v, t+1, e)
        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (b, v, t, e) -> (v, b, t, e)

        attn_mask = tf.concat([tf.ones((B, V, 1)), attn_mask], axis=-1)  # (b, v, t) -> (b, v, t+1)
        attn_mask = tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t+1) -> (v, b, t+1)

        X_enc = X
        X_enc = self.dropout(X_enc)
        for i in range(self.num_layers):
            # X_enc = self.encoders[i](X_enc, attention_mask=attn_mask)
            if i == self.num_layers - 1:
                X_enc = self.encoders[i](X_enc, attention_mask=None)
            else:
                X_enc = self.encoders[i](X_enc, attention_mask=attn_mask)

        X_enc = self.dropout(X_enc)
        cls_output = X_enc[:, :, 0]  # cls_output: (v, b, e)
        cls_output = tf.transpose(cls_output, perm=[1, 0, 2])  # cls_output: (b, v, e)
        # cls_output = self.seq(cls_output)  # cls_output: (b, v, e*2)
        cls_output_1 = cls_output[:, 0]
        pred = self.predictor(cls_output_1)

        # Auxiliary task: interpolation
        aux_mask = tf.concat([tf.zeros((B, V, 1)), aux_mask], axis=-1)  # (b, v, t) -> (b, v, t+1)
        aux_mask = tf.transpose(aux_mask, perm=[1, 0, 2])  # (b, v, t+1) -> (v, b, t+1)
        combined_mask = attn_mask * (1.0 - aux_mask)

        X_aux = X
        X_aux = self.dropout(X_aux)
        for i in range(self.num_layers):
            # X_aux = self.encoders[i](X_aux, attention_mask=combined_mask)
            if i == self.num_layers - 1:
                X_aux = self.encoders[i](X_aux, attention_mask=None)
            else:
                X_aux = self.encoders[i](X_aux, attention_mask=combined_mask)

        X_aux = self.dropout(X_aux)
        cls_output = X_aux[:, :, 0]  # cls_output: (v, b, e)
        aux_pred = self.aux_recn_head(cls_output)  # (v, b, e) -> (v, b, t)
        # aux_loss = tf.reduce_sum(aux_mask[:, :, 1:] * tf.square(aux_pred - aux_recn_true)) / tf.reduce_sum(aux_mask)
        aux_mask = aux_mask[:, :, 1:]
        aux_pred = tf.reduce_mean(aux_mask * aux_pred, axis=2)  # (v, b, t) -> (v, b)
        aux_recn_true = tf.reduce_mean(aux_mask * aux_recn_true, axis=2)  # (v, b, t) -> (v, b)
        aux_loss = tf.reduce_mean(tf.square(aux_pred - aux_recn_true))
        self.add_loss(aux_loss)

        return pred
