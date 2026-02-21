import numpy as np
import tensorflow as tf

from ists_.model.embedding import TemporalEmbedding
from ists_.model.rope import FeedForward
from ists_.model.rope import GlobalSelfAttention
from ists_.model.rope_multivariate import GlobalMVSelfAttention


def get_optimal_dff(d_model):
    # 1. Target SwiGLU ratio (approx 2.66x)
    hidden_dim = 4 * d_model
    hidden_dim = int(2 * hidden_dim / 3)
    
    # 2. Force alignment to 32 (GPUs sweet spot for small dims)
    multiple_of = 32
    
    # 3. Round up
    dff = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    
    return dff


class SwiGLUFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, dff=None, activation='swish', dropout_rate=0.1, kernel_regularizer=None,
                 pre_layernorm=False, rms_scaling=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = get_optimal_dff(d_model) if dff is None else dff
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = kernel_regularizer
        self.pre_layernorm = pre_layernorm
        self.rms_scaling = rms_scaling

        # 1. The Gate Projection (w1 in LLaMA)
        self.gate_proj = tf.keras.layers.Dense(
            self.dff, 
            kernel_regularizer=kernel_regularizer,
            name="gate_proj"
        )
        
        # 2. The Value Projection (w2 in LLaMA)
        self.val_proj = tf.keras.layers.Dense(
            self.dff, 
            kernel_regularizer=kernel_regularizer,
            name="val_proj"
        )
        
        # 3. The Down Projection (w3 in LLaMA)
        self.down_proj = tf.keras.layers.Dense(
            d_model, 
            kernel_regularizer=kernel_regularizer,
            name="down_proj"
        )
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        
        self.layer_norm = tf.keras.layers.LayerNormalization(rms_scaling=self.rms_scaling)

    def call(self, x, training=False):
        y = x
        
        # 1. Pre-Normalization
        if self.pre_layernorm: 
            y = self.layer_norm(y)

        # 2. SwiGLU Logic
        # Branch 1: Gate (Projection -> Activation)
        gate = self.activation(self.gate_proj(y))
        # Branch 2: Value (Projection only)
        value = self.val_proj(y)
        # Element-wise multiplication
        x_swiglu = gate * value
        
        # 3. Down-projection back to d_model
        x_ffn = self.down_proj(x_swiglu)
        
        # 4. Dropout
        x_ffn = self.dropout(x_ffn, training=training)

        # 5. Residual Connection
        x = self.add([x, x_ffn])

        # 6. Post-Normalization
        if not self.pre_layernorm: 
            x = self.layer_norm(x)
            
        return x


class EncoderLGAttROPELayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None,
                 pre_layernorm=False, rms_scaling=False, max_freq=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.pre_layernorm = pre_layernorm
        self.rms_scaling = rms_scaling
        self.max_freq = max_freq

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.loc_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling, max_freq=self.max_freq
        )

        self.glb_attn = GlobalWindowAttention(
            window_size=15,
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling, max_freq=self.max_freq
        )

        if activation == 'swiglu':
            self.ffn = SwiGLUFFN(
                d_model=d_model,
                dropout_rate=dropout_rate, **reg,
                pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling
            )
        else:
            self.ffn = FeedForward(
                d_model=d_model,
                dff=dff,
                activation=activation,
                dropout_rate=dropout_rate, **reg,
                pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling
            )


    def call(self, x, attention_mask=None):
        x_shape = tf.shape(x)
        b, v, t, e = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        if attention_mask is None:
            # attention_mask = tf.ones((v, b, t), dtype=tf.float32)
            attention_mask = tf.ones((b, v, t), dtype=tf.float32)

        # x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (b, v, t, e)
        # attention_mask = tf.transpose(attention_mask, perm=[1, 0, 2])  # attention_mask: (b, v, t)

        # Local Attention
        attn_mask_loc = tf.reshape(attention_mask, (b * v, 1, t))  # attention_mask: (b*v, 1, t)  KMask
        x = tf.reshape(x, (b * v, t, e))  # x: (b*v, t, e)
        x = self.loc_attn(x, mask=attn_mask_loc)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        # Global Attention
        attn_mask_glb = tf.reshape(attention_mask, (b, 1, v * t))  # attention_mask: (b, 1, v*t)  KMask
        # x = tf.reshape(x, (b, v * t, e))  # x: (b, v*t, e)
        x = self.glb_attn(x, mask=attn_mask_glb)
        # x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)

        x = self.ffn(x)

        # x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (v, b, t, e)
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
        self.pre_layernorm = kwargs.pop('pre_layernorm', False)
        self.rms_scaling = kwargs.pop('rms_scaling', False)
        self.max_freq = kwargs.pop('max_freq', 10000.0)
        shared_weights = kwargs.pop('shared_weights', False)
        self.static_feats_ids = kwargs.pop('static_feats_ids', None)
        print('Ignored kwargs in EncoderLGARope:', kwargs)
        kwargs = {}

        self.embedder = TemporalEmbedding(
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            activation=self.activation,
            l2_reg=self.l2_reg,
            time_features=self.time_features,
            pos_enc=False,  # Position encoding is handled by RoPE
            custom_embedding=3,
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

        new_encoder_layer = lambda: EncoderLGAttROPELayer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
            pre_layernorm=self.pre_layernorm, rms_scaling=self.rms_scaling, max_freq=self.max_freq,
            **kwargs,
        )
        if shared_weights:
            self.encoder_layers = [new_encoder_layer()] * self.num_layers
        else:
            self.encoder_layers = [new_encoder_layer() for _ in range(self.num_layers)]

        self.variable_embeddings = None
        self.ve_scale = tf.math.sqrt(tf.cast(self.d_model/2, tf.float32))

    """def build(self, input_shape):
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
        self.variable_embeddings = tf.constant(variable_embeddings)
        self.ve_proj = tf.keras.layers.Dense(self.d_model, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg), name='ve_proj')  #, activation=self.activation)"""

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

        # B, V, T = tf.shape(X)[0], self.V, self.T

        tt = tf.gather(X, self.time_features_ids, axis=-1)  # (b, v, t, time_f)
        X = tf.gather(X, self.raw_feature_ids, axis=-1)  # (b, v, t, f') only raw features

        # rotary_angles = self.freq_generator(self.get_freq_generator_input(tt))  # (b*v, 1, t, head_dim)  ##1
        # rotary_angles = self.freq_generator(tt)  ##2
        # rotary_angles = tf.concat([rotary_angles, rotary_angles], axis=-1)[:, tf.newaxis]  # (b*v, 1, t, head_dim)  ##2

        X = self.embedder(X, tt)  ##1
        # rotary_angles = tf.reshape(rotary_angles, (B, V, 1, T, -1))  # (b*v, 1, t, head_dim) -> (b, v, 1, t, head_dim)

        '''# variable_embeddings = self.variable_embeddings  ##1
        variable_embeddings = self.ve_proj(self.variable_embeddings)  # (1, V, 1, d_model)  ##2
        variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
        variable_embeddings = variable_embeddings * self.ve_scale  # scale
        X = X + variable_embeddings'''

        # X, attn_mask = tf.transpose(X, perm=[1, 0, 2, 3]), tf.transpose(attn_mask, perm=[1, 0, 2])  # (b, v, t+1, e) -> (v, b, t+1, e)
        if not self.pre_layernorm: X = self.layernorm(X)  # post-embedder
        X = self.dropout(X)

        for i in range(self.num_layers):
            # X = self.encoder_layers[i](X, rotary_angles=rotary_angles, attention_mask=attn_mask)
            X = self.encoder_layers[i](X, attention_mask=attn_mask)

        # X = tf.transpose(X, perm=[1, 0, 2, 3])  # (v, b, t+1, e) -> (b, v, t+1, e)
        if self.pre_layernorm: X = self.layernorm(X)  # pre-readout
        return X


class GlobalWindowAttention(GlobalMVSelfAttention):
    """
    Specializes GlobalSelfAttention by applying a sliding-window mask.
    """

    def __init__(self, window_size, **kwargs):
        # Initialize the parent (GlobalSelfAttention)
        super().__init__(**kwargs)
        self.window_size = window_size

    def create_sliding_window_mask(self):
        """Creates a boolean sliding window attention mask."""
        S = self.seq_len
        W = self.window_size
        if W % 2 == 0:
            raise ValueError("Window size must be an odd number.")
        q_indices = tf.range(S)
        k_indices = tf.range(S)
        distance_matrix = q_indices[:, tf.newaxis] - k_indices[tf.newaxis, :]
        half_window = W // 2
        return tf.abs(distance_matrix) <= half_window

    def build(self, input_shape):
        _, C, S, _ = input_shape
        self.channel, self.seq_len = C, S

        with tf.init_scope():
            # 1. Create the local sliding-window mask
            # local_mask = create_sliding_window_mask(S, self.window_size)  # (S, S)
            local_mask = self.create_sliding_window_mask()  # (S, S)

            # Expand to (C*S, C*S) for multi-channel input
            local_mask = tf.tile(local_mask, [C, C])  # (C*S, C*S)

            self.local_mask = tf.Variable(initial_value=tf.cast(local_mask, tf.float32),
                                          trainable=False, dtype=tf.float32, name="local_mask")

        super().build(input_shape)

    def call(self, x, mask=None):  # x: (B, C*S, E) attn_mask: (B, 1, C*S)
        local_mask = self.local_mask

        # 2. Combine with any external mask (e.g., padding mask) if provided
        if mask is not None:
            # Keras MHA masks are boolean. False means "mask this token".
            # We combine them with a logical AND. A token is attended to
            # only if it's valid in BOTH the local window AND the external mask.
            # The external mask needs to be reshaped to be broadcastable with the local mask.
            # External mask shape: (B, 1, C*S)
            # local_mask shape: (C*S, C*S) -> (1, C*S, C*S)
            # Combined shape will be broadcast to (B, C*S, C*S)
            combined_mask = local_mask[tf.newaxis, :, :] * mask  # (B, C*S, C*S)
        else:
            combined_mask = local_mask

        # 3. Call the parent's call method with the final mask
        return super().call(x, mask=combined_mask)
