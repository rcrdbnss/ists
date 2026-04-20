import numpy as np
import tensorflow as tf

import ists_.model.model
from ists_.model.embedding import PositionalEmbedding, TemporalEmbedding
from ists_.model.window_attention import GlobalWindowAttention


class PositionalEmbeddingCLS(PositionalEmbedding):

    def call(self, x):
        return self.pe[:, 1:1 + tf.shape(x)[-2], :]


class TemporalEmbeddingCLS(TemporalEmbedding):

    def __init__(self, d_model, kernel_size, time_features=None, activation="relu", l2_reg=None):
        super().__init__(d_model, kernel_size, False, time_features, activation, l2_reg)
        self.pos_embedder = PositionalEmbeddingCLS(self.d_model)

    def build(self, input_shape):
        super().build(input_shape)


def compute_masked_mean(x_raw, mask, epsilon=1e-6):
    masked_sum = tf.reduce_sum(x_raw * mask, axis=-1)
    masked_count = tf.reduce_sum(mask, axis=-1)
    mean = masked_sum / (masked_count + epsilon)  # Avoid division by zero
    return mean


class ISTEncoderCLS(tf.keras.Model):

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
        """value_ids = np.arange(len(feature_mask))
        arg_null = feature_mask == 1
        if arg_null.any():
            self.feature_mask, self.value_ids = feature_mask[~arg_null], value_ids[~arg_null]
            self.attn_mask_id = value_ids[arg_null][0]
        else:
            self.feature_mask, self.value_ids = feature_mask, value_ids
            self.attn_mask_id = None"""
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

        self.embedder = TemporalEmbeddingCLS(
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            time_features=self.time_features,
            activation=self.activation,
            l2_reg=self.l2_reg
        )

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)

        self.cls_token = self.add_weight(shape=(1, 1, 1, self.d_model), initializer="random_normal", trainable=True)

        encoder_layer_cls = MVEncoderLayerLGA  # todo: support for multiple encoder layer classes
        shared_weights = kwargs.get('shared_weights', False)
        new_encoder_layer = lambda: encoder_layer_cls(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg
            # lga_shared_weights=True,
        )
        if shared_weights:
            self.encoder_layers = [new_encoder_layer()] * self.num_layers
        else:
            self.encoder_layers = [new_encoder_layer() for _ in range(self.num_layers)]

        self.variable_embeddings = None
        self.last_layer_no_mask = kwargs.get('last_layer_no_mask', False)

    def build(self, input_shape):
        V = input_shape[0][1]

        """variable_embeddings = variable_embeddings_regular_simplex_dense(V, self.d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)"""

        pe = np.zeros((V, self.d_model), dtype=np.float32)
        position = np.expand_dims(np.arange(0, V), 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, 0)  # (1, V, d_model)
        variable_embeddings = pe[:, :, tf.newaxis, :]  # (1, V, 1, d_model)

        self.variable_embeddings = self.add_weight(
            shape=(1, V, 1, self.d_model),
            initializer=tf.keras.initializers.Constant(variable_embeddings),
            # trainable=True,
            trainable=False,
        )

        # self.variable_embeddings = tf.keras.layers.Embedding(V, self.d_model)

        """self.ve_scale = self.add_weight(
            name='ve_scale',
            shape=(),
            initializer='ones',
            trainable=True,
        )"""
        self.ve_scale = 1.0

    def call(self, inputs):  # (b, v, t, f)
        # exg_x, _ = inputs
        # X, attn_mask = exg_x
        X, attn_mask = inputs

        X_shape = tf.shape(X)
        B, V, T = X_shape[0], X_shape[1], X_shape[2]

        """if self.attn_mask_id is None:
            X, attn_mask = X, tf.ones((B, V, T), dtype=tf.float32)
        else:
            X, attn_mask = tf.gather(X, self.value_ids, axis=-1), tf.gather(X, self.attn_mask_id, axis=-1)"""

        X = tf.reshape(X, (V * B, T, -1))  # (b, v, t, f) -> (v*b, t, f)
        X = self.embedder(X)
        X = tf.reshape(X, (B, V, T, -1))  # (v*b, t, e) -> (b, v, t, e)

        cls_token = tf.tile(self.cls_token, [B, V, 1, 1])  # (1, 1, 1, e) -> (b, v, 1, e)
        cls_pe = self.embedder.pos_embedder.pe[:, 0:1, :][tf.newaxis]  # (1, 1, 1, e)
        cls_token = cls_token + cls_pe
        X = tf.concat([cls_token, X], axis=2)  # (b, v, 1, e) + (b, v, t, e) -> (b, v, t+1, e)

        variable_embeddings = self.variable_embeddings
        """variable_embeddings = tf.range(0, V)  # (V,)
        variable_embeddings = self.variable_embeddings(variable_embeddings)  # (V, d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)"""
        variable_embeddings = variable_embeddings * self.ve_scale  # (1, V, 1, e) -> (1, V, 1, e)
        X = X + variable_embeddings

        attn_mask = tf.concat([tf.ones((B, V, 1)), attn_mask], axis=-1)  # (b, v, t) -> (b, v, t+1)

        X, attn_mask = tf.transpose(X, perm=[1, 0, 2, 3]), tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t+1, e) -> (v, b, t+1, e)
        X = self.dropout(X)

        for i in range(self.num_layers):
            X = self.encoder_layers[i](X, attention_mask=attn_mask)
            """if self.last_layer_no_mask and i == self.num_layers - 1:
                # X = self.encoder_layers[i](X, attention_mask=attn_mask, do_global=False)
                # X = self.encoder_layers[i](X, do_global=False)
                X = self.encoder_layers[i](X)
            else:
                X = self.encoder_layers[i](X, attention_mask=attn_mask)"""

        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (v, b, t+1, e) -> (b, v, t+1, e)

        """X = self.dropout_1(X)
        cls_output = X[:, :, 0]  # cls_output: (v, b, e)
        cls_output = tf.transpose(cls_output, perm=[1, 0, 2])  # cls_output: (b, v, e)

        pred = self.forecasting_head(cls_output)  # (b, v, e) -> (b, v, 1)
        pred = pred[:, 0]  # first variable only, (b, 1)

        return pred"""

        return X


class ISTInterpolationCLS(tf.keras.Model):

    def __init__(self, encoder: ISTEncoderCLS):
        super().__init__()
        self.encoder = encoder
        self.raw_feature_id = encoder.raw_feature_id
        self.d_model = encoder.d_model
        self.dff = encoder.dff
        self.activation = encoder.activation
        self.dropout_rate = encoder.dropout_rate
        self.l2_reg = encoder.l2_reg

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.intp_head = Regressor(1, 0, self.activation, self.dropout_rate, 0, name='intp_head')
        self.intp_head = tf.keras.layers.Dense(1, activation='linear', name='intp_head')
        # self.intp_head = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))], name='intp_head')
        self.mse = tf.keras.metrics.Mean(name='mse')
        self.mae = tf.keras.metrics.Mean(name='mae')

        # self.mean_head = Regressor(1, 0, self.activation, self.dropout_rate, 0, name='mean_head')
        self.mean_head = tf.keras.layers.Dense(1, activation='linear', name='mean_head')
        # self.mean_head = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))], name='mean_head')
        self.mean_task_weight = 1.0  # weight for mean task loss
        # self.mean_task_weight = 0  # disable mean task

    def call(self, inputs):  # (b, v, t, f)
        # (X, attn_mask, intp_mask), others = inputs
        (X, attn_mask, X_aux, intp_mask), (X_spt, attn_mask_spt, X_spt_aux, intp_mask_spt) = inputs

        """X_spt = X_spt[:, 1:]  # the first variable is the target variable, already included in X
        attn_mask_spt = attn_mask_spt[:, 1:]
        X_spt_aux = X_spt_aux[:, 1:]
        intp_mask_spt = intp_mask_spt[:, 1:]
        # include spatial neighbors variables
        X = tf.concat([X, X_spt], axis=1)  # (b, v, t, f) + (b, v_spt, t, f) -> (b, v+v_spt, t, f)
        attn_mask = tf.concat([attn_mask, attn_mask_spt], axis=1)  # (b, v, t) + (b, v_spt, t) -> (b, v+v_spt, t)
        X_aux = tf.concat([X_aux, X_spt_aux], axis=1)  # (b, v, t, f) + (b, v_spt, t, f) -> (b, v+v_spt, t, f)
        intp_mask = tf.concat([intp_mask, intp_mask_spt], axis=1)  # (b, v, t) + (b, v_spt, t) -> (b, v+v_spt, t)"""

        X_orig = X
        X = X_aux

        X_raw = X_orig[:, :, :, self.raw_feature_id]  # (b, v, t)
        combined_mask = attn_mask * (1.0 - intp_mask)  # (b, v, t)
        mean_true = compute_masked_mean(X_raw, combined_mask)  # (b, v)

        # X = self.encoder(((X, combined_mask), others))  # (b, v, t+1, e)
        X = self.encoder((X, combined_mask))  # (b, v, t+1, e)
        X = self.dropout(X)

        cls_token = X[:, :, 0]  # cls_token: (b, v, e)
        X = X[:, :, 1:]  # remove cls token, (b, v, t, e)

        intp_pred = self.intp_head(X)  # (b, v, t, e) -> (b, v, t, 1)
        intp_pred = tf.squeeze(intp_pred, axis=-1)  # (b, v, t, 1) -> (b, v, t)
        # intp_mask = tf.transpose(intp_mask, perm=[1, 0, 2])  # (b, v, t) -> (v, b, t)
        intp_pred = intp_pred * intp_mask
        # X_raw = tf.transpose(X_raw, perm=[1, 0, 2])  # (b, v, t) -> (v, b, t)
        intp_true = X_raw * intp_mask
        intp_loss = tf.reduce_mean(tf.square(intp_pred - intp_true))
        int_mae = tf.reduce_mean(tf.abs(intp_pred - intp_true))
        self.mae.update_state(int_mae)
        self.mse.update_state(intp_loss)

        mean_pred = self.mean_head(cls_token)  # (b, v, e) -> (b, v, 1)
        mean_pred = tf.squeeze(mean_pred, axis=-1)  # (b, v, 1) -> (b, v)
        # mean_pred = tf.transpose(mean_pred, perm=[1, 0])  # (v, b) -> (b, v)
        mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))

        loss = intp_loss + (self.mean_task_weight * mean_loss if self.mean_task_weight > 0 else 0)
        self.add_loss(loss)

        return loss


class ISTForecastingCLS(tf.keras.Model):

    def __init__(self, encoder: ISTEncoderCLS):
        super().__init__()
        self.encoder = encoder
        self.raw_feature_id = encoder.raw_feature_id
        self.d_model = encoder.d_model
        self.num_heads = encoder.num_heads
        self.dff = encoder.dff
        self.activation = encoder.activation
        self.dropout_rate = encoder.dropout_rate
        self.l2_reg = encoder.l2_reg

        """self.encoder_layers = [MVEncoderLayerLGA(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            # lga_shared_weights=True
        ) for i in range(2)]"""

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        # self.pred_head = Regressor(1, 0, self.activation, self.dropout_rate, 0, name='pred_head')
        self.pred_head = tf.keras.layers.Dense(1, activation='linear', name='pred_head')
        # self.pred_head = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))], name='pred_head')

        # self.mean_head = Regressor(1, 0, self.activation, self.dropout_rate, 0, name='mean_head')
        self.mean_head = tf.keras.layers.Dense(1, activation='linear', name='mean_head')
        # self.mean_head = tf.keras.models.Sequential([tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))], name='mean_head')
        self.mean_task_weight = 1.0  # weight for mean task loss
        # self.mean_task_weight = 0  # disable mean task

        # self.pooler = AttentivePooling()

    def call(self, inputs):  # (b, v, t, f)
        # (X, attn_mask), others = inputs
        (X, attn_mask), (X_spt, attn_mask_spt) = inputs

        """X_spt = X_spt[:, 1:]  # the first variable is the target variable, already included in X
        attn_mask_spt = attn_mask_spt[:, 1:]
        # include spatial neighbors variables
        X = tf.concat([X, X_spt], axis=1)  # (b, v, t, f) + (b, v_spt, t, f) -> (b, v+v_spt, t, f)
        attn_mask = tf.concat([attn_mask, attn_mask_spt], axis=1)  # (b, v, t) + (b, v_spt, t) -> (b, v+v_spt, t)"""

        X_raw = X[:, :, :, self.raw_feature_id]  # (b, v, t)
        mean_true = compute_masked_mean(X_raw, attn_mask)  # (b, v)

        # X = self.encoder(((X, attn_mask), others))  # (b, v, t+1, e)
        X = self.encoder((X, attn_mask))  # (b, v, t+1, e)

        cls_token = self.dropout(X)[:, :, 0]
        mean_pred = self.mean_head(cls_token)  # (b, v, e) -> (b, v, 1)
        mean_pred = tf.squeeze(mean_pred, axis=-1)  # (b, v, 1) -> (b, v)
        mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))
        self.add_loss(mean_loss * self.mean_task_weight if self.mean_task_weight > 0 else tf.constant(0.0))

        """# additional encoder layer pass
        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (b, v, t+1, e) -> (v, b, t+1, e)
        for i in range(len(self.encoder_layers)):
            X = self.encoder_layers[i](X)  # todo: attention mask
        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (v, b, t+1, e) -> (b, v, t+1, e)"""

        X = self.dropout(X)

        cls_token = X[:, :, 0]  # cls_output: (b, v, e)
        """X = X[:, :, 1:]  # remove cls token, (b, v, t, e)
        X_shape = tf.shape(X)
        B, V, T, _ = X_shape[0], X_shape[1], X_shape[2], X_shape[3]
        X = tf.reshape(X, (B*V, T, self.d_model))  # (b, v, t, e) -> (b*v, t, e)
        pooled = self.pooler(X)  # (b*v, e)
        pooled = tf.reshape(pooled, (B, V, self.d_model))  # (b*v, e) -> (b, v, e)
        cls_token = tf.concat([cls_token, pooled], axis=-1)  # (b, v, e) + (b, v, e) -> (b, v, 2*e)"""

        """pred = self.forecasting_head(cls_token)  # (b, v, e) -> (b, v, 1)
        pred = pred[:, 0]  # first variable only"""
        pred = self.pred_head(cls_token[:, 0])  # first variable only, (b, e) -> (b, 1)

        """mean_pred = self.mean_head(cls_token)  # (b, v, e) -> (b, v, 1)
        mean_pred = tf.squeeze(mean_pred, axis=-1)  # (b, v, 1) -> (b, v)
        mean_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))
        self.add_loss(mean_loss * self.mean_task_weight if self.mean_task_weight > 0 else tf.constant(0.0))"""

        return pred

    '''def compile(self, encoder_optimizer, head_optimizer, **kwargs):
        """
        Custom compile method to accept two optimizers.
        """
        super().compile(**kwargs)
        self.encoder_optimizer = encoder_optimizer
        self.head_optimizer = head_optimizer

    def train_step(self, data):
        """
        Overrides the default training step to apply different learning rates.
        """
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute loss
            loss = self.compute_loss(x, y, y_pred)
            self._loss_tracker.update_state(
                loss, sample_weight=tf.shape(keras.src.tree.flatten(x)[0])[0],
            )
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        # Separate the trainable variables for each part of the model
        encoder_vars = self.encoder.trainable_variables + self.mean_head.trainable_variables
        head_vars = self.forecasting_head.trainable_variables

        # Calculate gradients for all variables at once
        all_trainable_vars = encoder_vars + head_vars
        grads = tape.gradient(loss, all_trainable_vars)

        # Split the gradients to match the variables
        encoder_grads = grads[:len(encoder_vars)]
        head_grads = grads[len(encoder_vars):]

        # Apply gradients using the respective optimizers
        if len(encoder_vars) > 0: self.encoder_optimizer.apply_gradients(zip(encoder_grads, encoder_vars))
        self.head_optimizer.apply_gradients(zip(head_grads, head_vars))

        # Update metrics
        return self.compute_metrics(x, y, y_pred)'''


class MVEncoderLayerLGA(ists_.model.model.EncoderLocalGlobalAttnMaskLayer):

    def __init__(self, *, lga_shared_weights=False, **kwargs):
        super().__init__(**kwargs)
        """if lga_shared_weights:
            del self.glb_attn
            self.glb_attn = self.loc_attn  # Use local attention for global attention as well"""
        del self.glb_attn
        self.attn_kwargs = {
            'num_heads': kwargs['num_heads'],
            'key_dim': kwargs['d_model'] // kwargs['num_heads'],
            'dropout': kwargs['dropout_rate'],
            'kernel_regularizer': tf.keras.regularizers.l2(kwargs['l2_reg']) if kwargs['l2_reg'] else None,
            'pre_layernorm': kwargs.get('pre_layernorm', False),
            'rms_scaling': kwargs.get('rms_scaling', False),
        }
        # self.cls_attn = CrossAttention(**self.attn_kwargs)

    def build(self, x_shape):
        B, V, T, E = x_shape
        self.glb_attn = GlobalWindowAttention(
            sequence_length=T, window_size=15, channels=V,
            **self.attn_kwargs
        )
        super().build(x_shape)

    def call(self, x, attention_mask=None, do_local=True, do_global=True):  # x: (b, v, t, e) attn_mask: (b, v, t)
        shape = tf.shape(x)
        b, v, t, e = shape[0], shape[1], shape[2], shape[3]

        attn_mask = attention_mask
        if attn_mask is None:
            attn_mask = tf.ones((b, v, t), dtype=tf.float32)

        if do_local:
            # attn_mask_loc = tf.reshape(attn_mask, (b*v, t, 1))  # attn_mask: (b*v, t, 1)  QMask
            attn_mask_loc = tf.reshape(attn_mask, (b*v, 1, t))  # attn_mask: (b*v, 1, t)  KMask
            """attn_mask_loc = tf.reshape(attn_mask, (b*v, t))  # attn_mask: (b*v, t)
            attn_mask_loc = tf.expand_dims(attn_mask_loc, -1) * tf.expand_dims(attn_mask_loc, 1)  # attn_mask: (b*v, t, t)  symmetric mask"""

            x = tf.reshape(x, (b*v, t, e))  # x: (b*v, t, e)
            x = self.loc_attn(x, attention_mask=attn_mask_loc)
            x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        if do_global:
            # attn_mask_glb = tf.reshape(attn_mask, (b, v*t, 1))  # attn_mask: (b, v*t, 1)  QMask
            attn_mask_glb = tf.reshape(attn_mask, (b, 1, v*t))  # attn_mask: (b, 1, v*t)  KMask
            """attn_mask_glb = tf.reshape(attn_mask, (b, v*t))  # attn_mask: (b, v*t)
            attn_mask_glb = tf.expand_dims(attn_mask_glb, -1) * tf.expand_dims(attn_mask_glb, 1)  # attn_mask: (b, v*t, v*t)  symmetric mask"""

            """# KMask+GAnoCLS CLS does not attend any token
            is_not_first_token = tf.cast(tf.range(t) > 0, dtype=tf.float32)
            is_not_first_token = is_not_first_token[tf.newaxis, tf.newaxis, :]  # (1, 1, t)
            attn_mask_glb = attn_mask * is_not_first_token  # (b, v, t)
            mul = tf.ones((b, v, t)) * is_not_first_token  # (b, v, t)
            attn_mask_glb = tf.reshape(attn_mask_glb, (b, 1, v * t))  # attn_mask: (b, 1, v*t)
            mul = tf.reshape(mul, (b, v * t, 1))  # mul: (b, v*t, 1)
            attn_mask_glb = mul * attn_mask_glb  # attn_mask_glb: (b, v*t, v*t)"""

            x = tf.reshape(x, (b, v*t, e))  # x: (b, v*t, e)
            x = self.glb_attn(x, attention_mask=attn_mask_glb)
            x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

            """# CLS token attention
            cls_token = x[:, :, 0:1, :]  # (b, v, 1, e)
            cls_token = tf.reshape(cls_token, (b*v, 1, e))  # (b*v, 1, e)
            x = tf.reshape(x, (b*v, t, e))  # (b*v, t, e)
            attn_mask_cls = tf.reshape(attn_mask, (b*v, 1, t))  # attn_mask: (b*v, 1, t)
            cls_token = self.cls_attn(cls_token, context=x, attention_mask=attn_mask_cls)
            cls_token = tf.reshape(cls_token, (b, v, 1, e))  # (b, v, 1, e)
            x = tf.reshape(x, (b, v, t, e))  # (b, v, t, e)
            x = tf.concat([cls_token, x[:, :, 1:]], axis=2)  # (b, v, t, e)"""

        x = self.ffn(x)

        return x


def Regressor(output_size, hidden_units=0, activation="relu", dropout_rate=0.1, l2_reg=None, name=None):
    l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    seq = []
    if hidden_units > 0:
        seq.extend([
            tf.keras.layers.Dense(hidden_units, activation=activation, kernel_regularizer=l2_reg),
            # tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ])
    seq.append(tf.keras.layers.Dense(output_size, activation='linear', kernel_regularizer=l2_reg))
    seq = tf.keras.models.Sequential(seq, name=name)
    return seq
