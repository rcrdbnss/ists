import numpy as np
import tensorflow as tf

from ists_.model.embedding import TemporalEmbedding, variable_embeddings_regular_simplex_dense
from ists_.model.model_cls import MVEncoderLayerLGA, compute_masked_mean


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

        feature_mask = np.array(feature_mask)
        self.feature_mask = feature_mask
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

        self.embedder = TemporalEmbedding(
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            feature_mask=self.feature_mask,
            time_features=self.time_features,
            activation=self.activation,
            l2_reg=self.l2_reg
        )
        self.layernorm = tf.keras.layers.LayerNormalization() #rms_scaling=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        encoder_layer_class = MVEncoderLayerLGA  # todo: support for multiple encoder layer classes
        shared_weights = kwargs.get('shared_weights', False)
        new_encoder_layer = lambda: encoder_layer_class(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg
        )
        if shared_weights:
            self.encoder_layers = [new_encoder_layer()] * self.num_layers
        else:
            self.encoder_layers = [new_encoder_layer() for _ in range(self.num_layers)]

        self.variable_embeddings = None
        self.ve_scale = 1.0

        self.static_feats_ids = kwargs.get('static_feats_ids', None)

    def build(self, input_shape):
        V = input_shape[0][1]

        variable_embeddings = variable_embeddings_regular_simplex_dense(V, self.d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)

        self.variable_embeddings = self.add_weight(
            shape=(1, V, 1, self.d_model),
            initializer=tf.keras.initializers.Constant(variable_embeddings),
            trainable=True,
            # trainable=False,
        )

        self.ve_scale = self.add_weight(
            name='ve_scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(tf.math.sqrt(tf.cast(self.d_model/2, tf.float32))), #'ones',
            # trainable=True,
            trainable=False,
        )

    def call(self, inputs):  # (b, v, t, f)
        # exg_x, _ = inputs
        # X, attn_mask = exg_x
        X, attn_mask, X_static = inputs

        X_shape = tf.shape(X)
        B, V, T = X_shape[0], X_shape[1], X_shape[2]

        X = tf.reshape(X, (V * B, T, -1))  # (b, v, t, f) -> (v*b, t, f)
        X = self.embedder(X)
        X = tf.reshape(X, (B, V, T, -1))  # (v*b, t, e) -> (b, v, t, e)


        variable_embeddings = self.variable_embeddings
        variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
        """variable_embeddings = tf.range(0, V)  # (V,)
        variable_embeddings = self.variable_embeddings(variable_embeddings)  # (V, d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)"""
        variable_embeddings = variable_embeddings * self.ve_scale  # (1, V, 1, e) -> (1, V, 1, e)
        X = X + variable_embeddings

        # attn_mask = tf.concat([tf.ones((B, V, 1)), attn_mask], axis=-1)  # (b, v, t) -> (b, v, t+1)

        X, attn_mask = tf.transpose(X, perm=[1, 0, 2, 3]), tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t+1, e) -> (v, b, t+1, e)
        X = self.layernorm(X)
        X = self.dropout(X)

        for i in range(self.num_layers):
            X = self.encoder_layers[i](X, attention_mask=attn_mask)

        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (v, b, t+1, e) -> (b, v, t+1, e)
        # X = self.layernorm(X)
        return X


class ISTInterpolationAttnPool(tf.keras.Model):

    def __init__(self, encoder: ISTEncoder):
        super().__init__()
        self.encoder = encoder
        self.raw_feature_id = encoder.raw_feature_id
        self.d_model = encoder.d_model
        self.dff = encoder.dff
        self.activation = encoder.activation
        self.dropout_rate = encoder.dropout_rate
        self.l2_reg = encoder.l2_reg

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.intp_head = Regressor(1, self.dff, self.activation, self.dropout_rate, self.l2_reg, name='intp_head')
        # self.intp_head = tf.keras.layers.Dense(1, activation='linear', name='intp_head', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.intp_head = ChannelWiseLinear(l2_reg=self.l2_reg)

        self.mse = tf.keras.metrics.Mean(name='mse')
        self.obs_mse = tf.keras.metrics.Mean(name='mse_obs')

        self.mean_pool = MeanPooling()
        # self.mean_head = Regressor(1, self.dff, self.activation, self.dropout_rate, self.l2_reg, name='mean_head')
        # self.mean_head = tf.keras.layers.Dense(1, activation='linear', name='mean_head', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.mean_head = ChannelWiseLinear(l2_reg=self.l2_reg)
        # self.mean_task_weight = 1.0  # weight for mean task loss
        self.mean_task_weight = 0.1
        # self.mean_task_weight = 0  # disable mean task
        if self.mean_task_weight > 0:
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
        (X_true, attn_mask, X_static, X, intp_mask), (X_spt_true, attn_mask_spt, X_spt_static, X_spt, intp_mask_spt) = inputs

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
        X_true, mean_true = self.drop_static_feats(X_true), self.drop_static_feats(mean_true)  # (b, v', t), (b, v')
        intp_mask = self.drop_static_feats(intp_mask)  # (b, v', t)

        X = self.encoder((X, combined_mask, X_static))  # (b, v, t, e)
        X = self.dropout(X)

        X_pred = self.intp_head(self.drop_static_feats(X))  # (b, v', t, e) -> (b, v', t, 1)
        X_pred = tf.squeeze(X_pred, axis=-1)  # (b, v', t, 1) -> (b, v', t)

        div = tf.reduce_sum(intp_mask) + 1e-9
        intp_mse = tf.reduce_sum(tf.square(X_pred - X_true) * intp_mask) / div
        self.mse.update_state(intp_mse)

        obs_mask = self.drop_static_feats(combined_mask)  # (b, v', t)
        div = tf.reduce_sum(obs_mask) + 1e-9
        obs_mse = tf.reduce_sum(tf.square(X_pred - X_true) * obs_mask) / div
        self.obs_mse.update_state(obs_mse)

        mean_pool = self.mean_pool(self.drop_static_feats(X), obs_mask)  # (b, v', t, e) -> (b, v', e)
        mean_pool = mean_pool[:, :, tf.newaxis, :]  # (b,v',e)->(b,v',1,e)
        mean_pred = self.mean_head(mean_pool)  # (b, v', e) -> (b, v', 1) or (b,v',1,e)->(b,v',1,1)
        mean_pred = mean_pred[:, :, 0, :]  # (b,v',1,1)->(b,v',1)
        mean_pred = tf.squeeze(mean_pred, axis=-1)  # (b, v', 1) -> (b, v')
        mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))

        # loss = intp_mse + (self.mean_task_weight * mean_loss if self.mean_task_weight > 0 else 0)
        # self.add_loss(loss)
        self.add_loss(intp_mse)
        self.add_loss(0.3 * obs_mse)
        loss = intp_mse + 0.3 * obs_mse
        if self.mean_task_weight > 0:
            self.aux_loss.update_state(mean_loss)
            loss += mean_loss * self.mean_task_weight
            self.add_loss(mean_loss * self.mean_task_weight)
        return X_pred


class ISTForecastingAttnPool(tf.keras.Model):

    def __init__(self, encoder: ISTEncoder):
        super().__init__()
        self.encoder = encoder
        self.raw_feature_id = encoder.raw_feature_id
        self.d_model = encoder.d_model
        self.num_heads = encoder.num_heads
        self.dff = encoder.dff
        self.activation = encoder.activation
        self.dropout_rate = encoder.dropout_rate
        self.l2_reg = encoder.l2_reg

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.pred_pool = AttentivePooling()
        self.pred_pool = MeanPooling()
        # self.pred_head = Regressor(1, self.dff, self.activation, self.dropout_rate, self.l2_reg, name='pred_head')
        self.pred_head = tf.keras.layers.Dense(1, activation='linear', name='pred_head', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))

        # self.mean_task_weight = 1.0  # weight for mean task loss
        self.mean_task_weight = 0  # disable mean task
        if self.mean_task_weight > 0:
            # self.mean_pool = AttentivePooling()
            self.mean_pool = MeanPooling()
            # self.mean_head = Regressor(1, self.dff, self.activation, self.dropout_rate, self.l2_reg, name='mean_head')
            # self.mean_head = tf.keras.layers.Dense(1, activation='linear', name='mean_head', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
            self.mean_head = ChannelWiseLinear(l2_reg=self.l2_reg)
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
        # (X, attn_mask), others = inputs
        (X, attn_mask, X_static), (X_spt, attn_mask_spt, X_spt_static) = inputs

        """X_spt = X_spt[:, 1:]  # the first variable is the target variable, already included in X
        attn_mask_spt = attn_mask_spt[:, 1:]
        # include spatial neighbors variables
        X = tf.concat([X, X_spt], axis=1)  # (b, v, t, f) + (b, v_spt, t, f) -> (b, v+v_spt, t, f)
        attn_mask = tf.concat([attn_mask, attn_mask_spt], axis=1)  # (b, v, t) + (b, v_spt, t) -> (b, v+v_spt, t)"""

        X_raw = X[:, :, :, self.raw_feature_id]  # (b, v, t)
        mean_true = compute_masked_mean(X_raw, attn_mask)  # (b, v)

        # X = self.encoder(((X, attn_mask), others))  # (b, v, t+1, e)
        X = self.encoder((X, attn_mask, X_static))  # (b, v, t, e)

        X = self.dropout(X)

        if self.mean_task_weight > 0:
            mean_pool = self.mean_pool(self.drop_static_feats(X), self.drop_static_feats(attn_mask))  # (b,v',t,e)->(b,v',e)
            mean_pool = mean_pool[:, :, tf.newaxis, :]  # (b,v',e)->(b,v',1,e)
            mean_pred = self.mean_head(mean_pool)  # (b, v', e) -> (b, v', 1) or (b,v',1,e)->(b,v',1,1)
            mean_pred = mean_pred[:, :, 0, :]  # (b,v',1,1)->(b,v',1)
            mean_pred = tf.squeeze(mean_pred, axis=-1)  # (b, v', 1) -> (b, v')
            mean_true = self.drop_static_feats(mean_true)
            mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))
            self.aux_loss.update_state(mean_loss)
            self.add_loss(mean_loss * self.mean_task_weight)

        X = X[:, 0]  # target variable only (b, v, t, e) -> (b, t, e)
        # X_pool = self.pred_pool(X, attn_mask[:, 0])  # (b, t, e) -> (b, e)
        X_pool = self.pred_pool(X)  # (b, t, e) -> (b, e)
        pred = self.pred_head(X_pool)  # (b, e) -> (b, 1)

        return pred


class MeanPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            inputs *= mask_float
            sum_inputs = tf.reduce_sum(inputs, axis=-2)
            count = tf.reduce_sum(mask_float, axis=-2)
            avg = sum_inputs / (count + 1e-9)
        else:
            avg = tf.reduce_mean(inputs, axis=-2)
        return avg


class AttentivePooling(tf.keras.layers.Layer):
    """Attentive Pooling implementation using a Dense layer."""

    def __init__(self, **kwargs):
        super(AttentivePooling, self).__init__(**kwargs)
        self.scorer = None
        self.scale = None

    def build(self, input_shape):
        # input_shape is (batch, seq_len, d_model)
        d_model = input_shape[-1]
        
        # Define the scaling factor: 1 / sqrt(d_model)
        self.scale = tf.math.rsqrt(tf.cast(d_model, tf.float32))

        # The query vector (kernel)
        self.scorer = tf.keras.layers.Dense(units=1, activation=None, name="attention_scorer", use_bias=False)
        super(AttentivePooling, self).build(input_shape)

    def call(self, inputs, mask=None):
        # 1. Calculate raw dot products
        # Shape: (batch, seq_len, 1)
        scores = self.scorer(inputs)

        # 2. APPLY SCALING (Critical for wider models)
        scores = scores * self.scale

        # 3. Apply mask
        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            scores += (1.0 - mask_float) * -1e9

        # 4. Softmax
        attention_weights = tf.nn.softmax(scores, axis=-2)

        # 5. Weighted Average
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=-2)

        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class ChannelWiseLinear(tf.keras.layers.Layer):
    def __init__(self, *, use_bias=True, l2_reg=None, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.l2_reg = None if l2_reg is None else tf.keras.regularizers.l2(l2_reg)

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
            initializer="glorot_uniform",
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
