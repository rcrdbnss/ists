import numpy as np
import tensorflow as tf

from ists_.model.embedding import centered_unit_simplex_embeddings, TemporalEmbedding, AnglesEmbedding, \
    centered_unit_simplex
from ists_.model.rope import FeedForward, apply_rotary_pos_emb
from ists_.model.rope import RoPEMultiHeadAttention, GlobalSelfAttention
from ists_.preprocessing import TIME_N_VALUES


class EncoderLGAttROPELayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None, **kwargs):
        super().__init__()
        self.pre_layernorm = kwargs.get('pre_layernorm', False)
        self.rms_scaling = kwargs.get('rms_scaling', False)

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.loc_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling
        )

        self.glb_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate, **reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling
        )

    def build(self, input_shape):
        v, b, t, e = input_shape
        self.V, self.T, self.E = v, t, e

    def call(self, x, attention_mask=None):
        v, b, t, e = self.V, tf.shape(x)[1], self.T, self.E

        if attention_mask is None:
            attention_mask = tf.ones((v, b, t), dtype=tf.float32)

        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (b, v, t, e)
        attention_mask = tf.transpose(attention_mask, perm=[1, 0, 2])  # attention_mask: (b, v, t)

        # Local Attention
        attn_mask_loc = tf.reshape(attention_mask, (b * v, t))  # attention_mask: (b*v, 1, t)  KMask
        x = tf.reshape(x, (b * v, t, e))  # x: (b*v, t, e)
        x = self.loc_attn(x, mask=attn_mask_loc)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        # Global Attention
        attn_mask_glb = tf.reshape(attention_mask, (b, v * t))  # attention_mask: (b, 1, v*t)  KMask
        x = tf.reshape(x, (b, v * t, e))  # x: (b, v*t, e)
        x = self.glb_attn(x, mask=attn_mask_glb)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        x = self.ffn(x)

        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (v, b, t, e)
        return x


class EncoderLGARope(tf.keras.Model):

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
            activation=self.activation,
            l2_reg=self.l2_reg,
            time_features=self.time_features,
            pos_enc=False,  # Position encoding is handled by RoPE
            custom_embedding=3
        )
        '''self.embedder = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        )'''
        self.layernorm = tf.keras.layers.LayerNormalization(rms_scaling=self.rms_scaling)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        '''self.freq_generator = PhysicalFreqGenerator(
            head_dim=d_model // num_heads,
            feature_configs=(
                [{'name': 'Pos', 'K': 100, 'type': 'linear'}] +
                ([{'name': f, 'K': TIME_N_VALUES[f], 'type': 'cyclical'} for f in time_features] if time_features else [])
            )
        )'''
        '''self.freq_generator = AnglesEmbedding(d_model // num_heads, time_features={
            f: TIME_N_VALUES[f] for f in time_features
        }, base=1000)'''

        shared_weights = kwargs.get('shared_weights', False)
        new_encoder_layer = lambda: EncoderLGAttROPELayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling
        )
        if shared_weights:
            self.encoder_layers = [new_encoder_layer()] * self.num_layers
        else:
            self.encoder_layers = [new_encoder_layer() for _ in range(self.num_layers)]

        self.variable_embeddings = None
        self.ve_scale = tf.math.sqrt(tf.cast(self.d_model/2, tf.float32))

        self.static_feats_ids = kwargs.get('static_feats_ids', None)

    def build(self, input_shape):
        input_shape = input_shape[0]
        B, V, T, _ = input_shape
        self.B, self.V, self.T = B, V, T

        '''variable_embeddings = centered_unit_simplex_embeddings(V, self.d_model)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)
        self.variable_embeddings = self.add_weight(
            shape=(1, V, 1, self.d_model),
            initializer=tf.keras.initializers.Constant(variable_embeddings),
            trainable=True,
            # trainable=False,
        )'''

        variable_embeddings = centered_unit_simplex(V)
        variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, V)
        self.ve_proj = tf.keras.layers.Dense(self.d_model, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='ve_proj')
        self.variable_embeddings = tf.constant(variable_embeddings)


    def get_freq_generator_input(self, tt):
        tt_shape = tf.shape(tt)
        B, T = tt_shape[0], tt_shape[1]
        pos_seq = tf.range(T, dtype=tf.float32)
        pos_seq = tf.expand_dims(pos_seq, 0)  # Batch dim
        pos_seq = tf.expand_dims(pos_seq, -1)  # Feature dim
        pos_seq = tf.broadcast_to(pos_seq, [B, T, 1])
        freq_in = tf.concat([pos_seq, tt], axis=-1)  # (B, T, n_features)
        return freq_in

    def call(self, inputs):  # (b, v, t, f)
        X, attn_mask, X_static = inputs

        B, V, T = tf.shape(X)[0], self.V, self.T

        X = tf.reshape(X, (B * V, T, -1))  # (b, v, t, f) -> (b*v, t, f)
        tt = tf.gather(X, self.time_features_ids, axis=-1)  # (b*v, t, time_f)
        X = tf.gather(X, self.raw_feature_ids, axis=-1)  # (b*v, t, f') only raw features

        # rotary_angles = self.freq_generator(self.get_freq_generator_input(tt))  # (b*v, 1, t, head_dim)  ##1
        # rotary_angles = self.freq_generator(tt)  ##2
        # rotary_angles = tf.concat([rotary_angles, rotary_angles], axis=-1)[:, tf.newaxis]  # (b*v, 1, t, head_dim)  ##2

        X = self.embedder(X, tt)  ##1
        # X = self.embedder(X) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))  ##2
        X = tf.reshape(X, (B, V, T, -1))  # (b*v, t, e) -> (b, v, t, e)
        # rotary_angles = tf.reshape(rotary_angles, (B, V, 1, T, -1))  # (b*v, 1, t, head_dim) -> (b, v, 1, t, head_dim)

        # variable_embeddings = self.variable_embeddings  ##1
        variable_embeddings = self.ve_proj(self.variable_embeddings)  # (1, V, 1, d_model)  ##2
        variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
        variable_embeddings = variable_embeddings * self.ve_scale  # scale
        X = X + variable_embeddings

        X, attn_mask = tf.transpose(X, perm=[1, 0, 2, 3]), tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t+1, e) -> (v, b, t+1, e)
        if not self.pre_layernorm: X = self.layernorm(X)  # post-embedder
        X = self.dropout(X)

        for i in range(self.num_layers):
            # X = self.encoder_layers[i](X, rotary_angles=rotary_angles, attention_mask=attn_mask)
            X = self.encoder_layers[i](X, attention_mask=attn_mask)

        X = tf.transpose(X, perm=[1, 0, 2, 3])  # (v, b, t+1, e) -> (b, v, t+1, e)
        if self.pre_layernorm: X = self.layernorm(X)  # pre-readout
        return X