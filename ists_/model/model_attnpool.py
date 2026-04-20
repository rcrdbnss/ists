import numpy as np
import tensorflow as tf

from ists_.model.embedding import TemporalEmbedding
from ists_.model.model_cls import MVEncoderLayerLGA, compute_masked_mean
from ists_.model.pooling import MeanPooling, AttentivePooling, LastTokenPooling, \
    AttentivePoolingWithPositionalBias


def heads_kernel_initializer(d_model, epsilon=0.01):
    limit = (3 * epsilon / d_model) ** 0.5
    return tf.keras.initializers.RandomUniform(-limit, +limit)


class ISTEncoder(tf.keras.Model):

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

        self.feature_mask = np.array(feature_mask)
        self.raw_feature_ids = np.where(self.feature_mask == 0)[0].tolist()
        self.time_features_ids = np.where(self.feature_mask == 2)[0].tolist()

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
        self.pre_layernorm = kwargs.get('pre_layernorm', False)
        self.rms_scaling = kwargs.get('rms_scaling', False)

        self.embedder = TemporalEmbedding(
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            time_features=self.time_features,
            activation=self.activation,
            l2_reg=self.l2_reg,
            custom_embedding=3,
        )
        self.layernorm = tf.keras.layers.RMSNormalization() if self.rms_scaling else tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        encoder_layer_class = MVEncoderLayerLGA  # todo: support for multiple encoder layer classes
        shared_weights = kwargs.get('shared_weights', False)
        new_encoder_layer = lambda: encoder_layer_class(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling,
        )
        if shared_weights:
            self.encoder_layers = [new_encoder_layer()] * self.num_layers
        else:
            self.encoder_layers = [new_encoder_layer() for _ in range(self.num_layers)]

        self.static_feats_ids = kwargs.get('static_feats_ids', None)

    def call(self, inputs):  # (b, v, t, f)
        X, attn_mask, tt, X_static = inputs

        X = self.embedder(X, tt)  # (b, v, t, e)

        if not self.pre_layernorm: X = self.layernorm(X)
        X = self.dropout(X)

        for i in range(self.num_layers):
            X = self.encoder_layers[i](X, attention_mask=attn_mask)

        if self.pre_layernorm: X = self.layernorm(X)
        return X


class ISTInterpolation(tf.keras.Model):

    def __init__(self, encoder: ISTEncoder):
        super().__init__()
        self.encoder = encoder
        self.raw_feature_id = encoder.raw_feature_ids[0]
        self.d_model = encoder.d_model
        self.dff = encoder.dff
        self.activation = encoder.activation
        self.dropout_rate = encoder.dropout_rate
        self.l2_reg = encoder.l2_reg

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.intp_head = Regressor(1, self.dff, self.activation, self.dropout_rate, self.l2_reg, name='intp_head')
        # self.intp_head = tf.keras.layers.Dense(1, activation='linear', name='intp_head', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.intp_head = ChannelWiseLinear(l2_reg=self.l2_reg, kernel_initializer=heads_kernel_initializer(self.d_model, 1))

        self.mse = tf.keras.metrics.Mean(name='mse')
        self.obs_weight = 0.3
        self.obs_mse = tf.keras.metrics.Mean(name='mse_obs')

        # self.mean_task_weight = 1.0  # weight for mean task loss
        self.mean_task_weight = 0.1
        # self.mean_task_weight = 0  # disable mean task
        if self.mean_task_weight > 0:
            self.mean_pool = MeanPooling()
            # self.mean_head = Regressor(1, self.dff, self.activation, self.dropout_rate, self.l2_reg, name='mean_head')
            # self.mean_head = tf.keras.layers.Dense(1, activation='linear', name='mean_head', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
            self.mean_head = ChannelWiseLinear(l2_reg=self.l2_reg, kernel_initializer=heads_kernel_initializer(self.d_model, 1))
            self.aux_loss = tf.keras.metrics.Mean(name='mse_avg')

    def build(self, input_shape):
        X_shape = input_shape[0][0]
        V = X_shape[1]
        if self.encoder.static_feats_ids is None:
            self.drop_static_feats = lambda X: X
        else:
            self.drop_static_feats = lambda X: tf.gather(
                X, [i for i in range(V) if i not in self.encoder.static_feats_ids], axis=1)

    def call(self, inputs):  # (b, v, t, f)
        (X_true, attn_mask, tt, X_static, X, intp_mask), (X_spt_true, attn_mask_spt, X_spt_static, X_spt, intp_mask_spt) = inputs

        """X_spt_true = X_spt_true[:, 1:]  # the first variable is the target variable, already included in X
        attn_mask_spt = attn_mask_spt[:, 1:]
        X_spt = X_spt[:, 1:]
        intp_mask_spt = intp_mask_spt[:, 1:]
        # include spatial neighbors variables
        X_true = tf.concat([X_true, X_spt_true], axis=1)  # (b, v, t, f) + (b, v_spt, t, f) -> (b, v+v_spt, t, f)
        attn_mask = tf.concat([attn_mask, attn_mask_spt], axis=1)  # (b, v, t) + (b, v_spt, t) -> (b, v+v_spt, t)
        X = tf.concat([X, X_spt], axis=1)  # (b, v, t, f) + (b, v_spt, t, f) -> (b, v+v_spt, t, f)
        intp_mask = tf.concat([intp_mask, intp_mask_spt], axis=1)  # (b, v, t) + (b, v_spt, t) -> (b, v+v_spt, t)"""

        X_true = X_true[:, :, :, self.raw_feature_id]  # (b, v, t)
        combined_mask = attn_mask * (1.0 - intp_mask)  # (b, v, t)
        mean_true = compute_masked_mean(X_true, combined_mask)  # (b, v)

        X = self.encoder((X, combined_mask, tt, X_static))  # (b, v, t, e)
        X = self.dropout(X)

        X_pred = self.intp_head(self.drop_static_feats(X))  # (b, v', t, e) -> (b, v', t, 1)
        X_pred = tf.squeeze(X_pred, axis=-1)  # (b, v', t, 1) -> (b, v', t)

        X_true = self.drop_static_feats(X_true)  # (b, v', t)
        intp_mask = self.drop_static_feats(intp_mask)  # (b, v', t)  # assuming static features are always present (attn_mask=1, intp_mask=0)
        div = tf.reduce_sum(intp_mask) + 1e-9
        intp_mse = tf.reduce_sum(tf.square(X_pred - X_true) * intp_mask) / div
        self.mse.update_state(intp_mse)

        combined_mask = self.drop_static_feats(combined_mask)  # (b, v', t)
        div = tf.reduce_sum(combined_mask) + 1e-9
        obs_mse = tf.reduce_sum(tf.square(X_pred - X_true) * combined_mask) / div
        self.obs_mse.update_state(obs_mse)

        obs_weight = self.obs_weight
        self.add_loss(intp_mse)
        self.add_loss(obs_weight * obs_mse)
        loss = intp_mse + obs_weight * obs_mse

        if self.mean_task_weight > 0:
            mean_true = self.drop_static_feats(mean_true)  # (b, v')
            mean_pool = self.mean_pool(self.drop_static_feats(X), combined_mask)  # (b, v', t, e) -> (b, v', e)
            mean_pool = mean_pool[:, :, tf.newaxis, :]  # (b,v',e)->(b,v',1,e)
            mean_pred = self.mean_head(mean_pool)  # (b, v', e) -> (b, v', 1) or (b,v',1,e)->(b,v',1,1)
            mean_pred = mean_pred[:, :, 0, :]  # (b,v',1,1)->(b,v',1)
            mean_pred = tf.squeeze(mean_pred, axis=-1)  # (b, v', 1) -> (b, v')
            mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))
            self.aux_loss.update_state(mean_loss)
            loss += mean_loss * self.mean_task_weight
            self.add_loss(mean_loss * self.mean_task_weight)

        return X_pred


class ISTForecasting(tf.keras.Model):

    def __init__(self, encoder: ISTEncoder, pooling='attn'):
        super().__init__()
        self.encoder = encoder
        self.l2_reg = encoder.l2_reg
        self.pooling = {
            'attn': AttentivePooling,
            'attn_pos_bias': AttentivePoolingWithPositionalBias,
            'last': LastTokenPooling,
            'mean': MeanPooling,
        }[pooling]()
        self.dropout = tf.keras.layers.Dropout(encoder.dropout_rate)
        l2_reg = None if self.l2_reg is None else tf.keras.regularizers.l2(self.l2_reg)
        self.head = tf.keras.layers.Dense(
            1, activation='linear', name='pred_head', kernel_regularizer=l2_reg,
            kernel_initializer=heads_kernel_initializer(self.encoder.d_model, 1)
        )

    def call(self, inputs):  # (b, v, t, f)
        (X, attn_mask, tt, X_static), _ = inputs

        X = self.encoder((X, attn_mask, tt, X_static))  # (b, v, t, e)

        X = self.dropout(X)

        X = X[:, 0]  # target variable only (b, v, t, e) -> (b, t, e)
        X_pool = self.pooling(X)  # (b, t, e) -> (b, e)
        pred = self.head(X_pool)  # (b, e) -> (b, 1)

        return pred


class ChannelWiseLinear(tf.keras.layers.Layer):
    def __init__(self, *, use_bias=True, l2_reg=None, kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.l2_reg = None if l2_reg is None else tf.keras.regularizers.l2(l2_reg)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        # input_shape: (batch, C, S, E)
        _, C, _, E = input_shape

        # store for reference
        self.channels = C
        self.emb_dim = E

        # one (E→1) weight per channel
        self.w = self.add_weight(
            name="channel_weights",
            shape=(C, E, 1),
            initializer=self.kernel_initializer,
            trainable=True,
            regularizer=self.l2_reg,
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="channel_bias",
                shape=(C, 1),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.b = None

    def call(self, x):
        # x: (B, C, S, E)
        y = tf.einsum("bcse,ceo->bcso", x, self.w)
        if self.b is not None:
            y = y + self.b[:, tf.newaxis, :]
        return y  # (B, C, S, 1)

    def compute_output_shape(self, input_shape):
        # Input: (B, C, S, E)
        return (input_shape[0], input_shape[1], input_shape[2], 1)
