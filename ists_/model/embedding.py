import numpy as np
import tensorflow as tf

from ists_.preprocessing import TIME_N_VALUES


class _PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, base=10000.0):
        super().__init__()

        # Compute the positional encodings once in log space.
        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))

        pe = np.concatenate([np.sin(position * div_term), np.cos(position * div_term)], axis=1)

        pe = np.expand_dims(pe, 0)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return self.pe[:, :tf.shape(x)[1], :]


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.pe = None

    def build(self, input_shape):
        if self.pe is None:
            with tf.init_scope():
                # Compute the positional encodings once in log space.
                position = np.expand_dims(np.arange(0, self.max_len), 1)
                div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(self.base) / self.d_model))
                pe = np.concatenate([np.sin(position * div_term), np.cos(position * div_term)], axis=1)
                pe = pe[np.newaxis].astype(np.float32)
                self.pe = tf.constant(pe, dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        return self.pe[:, :tf.shape(x)[1], :]


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model, base=10000.0):
        super(FixedEmbedding, self).__init__()

        # Create the embedding matrix
        position = np.expand_dims(np.arange(0, c_in), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))

        w = np.concatenate([np.sin(position * div_term), np.cos(position * div_term)], axis=1)

        # Initialize the embedding layer with the precomputed weights
        self.emb = tf.keras.layers.Embedding(c_in, d_model, embeddings_initializer=tf.constant_initializer(w),
                                             trainable=False)

    def call(self, x):
        return self.emb(x)


class TrainablePeriodicEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        """
        c_in: The Period T (e.g., 24 for hours, 7 for days). 
              Acts as the vocabulary size.
        d_model: The output dimension (must be even).
        """
        super(TrainablePeriodicEmbedding, self).__init__()
        self.d_model = d_model
        self.c_in = float(c_in)
        self.d_half = d_model // 2
        
        # --- 1. HARMONIC INITIALIZATION ---
        # Instead of Geometric (10000^...), we use Arithmetic Harmonics (1, 2, ... d/2)
        # distinct_positions: [0, 1, ..., c_in-1] -> Shape (c_in, 1)
        position = np.expand_dims(np.arange(0, c_in), 1)
        
        # frequencies: [1, 2, ..., d_half] -> Shape (1, d_half)
        harmonics = np.expand_dims(np.arange(1, self.d_half + 1), 0)
        
        # Compute angles: (2 * pi * pos * harmonic) / period
        # This ensures pos=0 and pos=c_in are mathematically identical in phase
        angles = (position * harmonics * 2 * np.pi) / self.c_in
        
        # Create sin/cos pairs
        w = np.concatenate([np.sin(angles), np.cos(angles)], axis=1) # Shape (c_in, d_model)

        # --- 2. TRAINABLE EMBEDDING LAYER ---
        # Initialize with our harmonics, but allow gradient descent to shift phases/amplitudes
        self.emb = tf.keras.layers.Embedding(
            input_dim=int(c_in), 
            output_dim=d_model,
            embeddings_initializer=tf.constant_initializer(w),
            trainable=True 
        )
        
        # Constant scaling factor to preserve vector length
        self.scale_factor = tf.sqrt(tf.cast(self.d_half, tf.float32))

    def call(self, x):
        # x: indices [Batch, Seq]
        
        # 1. Retrieve Embeddings
        # Shape: [Batch, Seq, d_model]
        vectors = self.emb(x)
        
        # 2. NORMALIZE & SCALE (Your Requirement)
        # Force vectors to lie on the hypersphere of radius sqrt(d_model/2)
        # This prevents the model from exploding weights to minimize loss
        vectors = tf.linalg.l2_normalize(vectors, axis=-1)
        
        return vectors * self.scale_factor


'''def cyclical_encoding(length, depth):
    depth = depth // 2
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)

    log_min = np.log(1.0 / length)
    log_max = np.log(0.5)

    angle_rates = np.linspace(log_max, log_min, depth)
    angle_rads = 2 * np.pi * positions * angle_rates  # (pos, depth)

    encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return encoding'''


def cyclical_encoding(length):
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    
    angle_rate = 1 / length
    angle_rads = 2 * np.pi * positions * angle_rate  # (pos, 1)

    encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)  # (pos, 2)

    return encoding


class CyclicalEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in):
        super().__init__()

        w = cyclical_encoding(c_in)

        # Initialize the embedding layer with the precomputed weights
        self.emb = tf.keras.layers.Embedding(
            input_dim=c_in,
            output_dim=2,
            embeddings_initializer=tf.constant_initializer(w),
            trainable=False
        )

    def call(self, x):
        return self.emb(x)


# --- Time Embedding (Unified Lookup) ---
class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, time_features):
        super().__init__()
        self.time_features = time_features  # {name: c_in}

        offsets = []
        curr_offset = 0
        weights_list = []
        for f, c_in in time_features.items():
            offsets.append(curr_offset)
            weights_list.append(cyclical_encoding(c_in))
            curr_offset += c_in

        # Offsets for broadcasting: (1, 1, F)
        self.time_offsets = tf.constant(offsets, dtype=tf.int32)[None, None, :]

        # Unified embedding table
        unified_weights = np.concatenate(weights_list, axis=0)
        self.time_lookup = tf.keras.layers.Embedding(
            input_dim=curr_offset,
            output_dim=2,  # Each feature gets sin/cos (2 dims)
            embeddings_initializer=tf.constant_initializer(unified_weights),
            trainable=False,
            name="time_embedding_lookup"
        )

    def build(self, input_shape):
        _, T, _ = input_shape
        self.T = T

    def call(self, tt):
        T, F = self.T, len(self.time_features)

        # Broadcast offsets and lookup
        # tt (B, T, F) + offsets (1, 1, F)
        tt_lookup = tf.cast(tt, tf.int32) + self.time_offsets
        t_emb = self.time_lookup(tt_lookup)  # (B, T, F, 2)

        # Flatten the last two dims: F features * 2 dims each
        t_emb = tf.reshape(t_emb, (-1, T, F*2))  # (B, T, F*2)
        return t_emb


class CustomPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, time_features=None, max_len=5000, base=10000.0):
        super().__init__()
        self.pos_embedder = PositionalEmbedding(d_model - 2 * len(time_features), max_len=max_len, base=base)
        self.time_features = {} if time_features is None else time_features  # {name: c_in}
        '''if self.time_features:
            self.time_embedders = [CyclicalEmbedding(c_in=c_in) for c_in in self.time_features.values()]'''
        # --- 2. Time Embedding (Unified Lookup) ---
        if self.time_features:
            self.time_lookup = TimeEmbedding(self.time_features)

    def call(self, tt):
        tt_shape = tf.shape(tt)
        B, C, T = tt_shape[0], tt_shape[1], tt_shape[2]

        pos_emb = self.pos_embedder(tf.reshape(tt, (B * C, T, -1)))  # (1, T, P)
        pos_emb = [tf.broadcast_to(pos_emb[tf.newaxis], [B, C, T, tf.shape(pos_emb)[-1]])]  # (B, C, T, P)

        '''if self.time_features:
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(tf.gather(tt, i, axis=-1))
                pos_emb.append(time_emb)'''
        # --- B. Process Time ---
        if self.time_features:
            t_emb = self.time_lookup(tt)  # (B, C, T, F*2)
            pos_emb.append(t_emb)
        pos_emb = tf.concat(pos_emb, axis=-1)

        return pos_emb


# --- NEW MERGED CLASS ---
class AnglesEmbedding(tf.keras.layers.Layer):
    """
    Computes a combined tensor of angles (radians) for positional encoding.

    Output Shape: (Batch, Seq, d_model // 2)

    The output consists of:
    1. Standard geometric progression angles (filling the dimensions not used by time features).
    2. Cyclical angles based on provided time features (1 angle per feature).
    """

    def __init__(self, d_model, time_features=None, max_len=5000, base=10000.0):
        super().__init__()
        self.d_model = d_model
        # time_features is expected to be {feature_name: period_length}
        self.time_features = time_features if time_features else {}

        # Total angles needed = d_model / 2 (since we will apply sin and cos later)
        total_angles = d_model // 2
        n_time_angles = len(self.time_features)
        n_pos_angles = total_angles - n_time_angles

        if n_pos_angles < 0:
            raise ValueError(f"d_model/2 ({total_angles}) is too small to hold {n_time_angles} time features.")

        # --- 1. Precompute Standard Positional Angles ---
        # Logic: angles = pos / (base ** (2i / d_model_subset))
        # Note: We use 2*n_pos_angles in the denominator to match standard transformer scale
        # for the reserved subspace.

        position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)

        # Frequencies for the geometric progression
        # We generate 'n_pos_angles' frequencies
        div_term = np.exp(np.arange(0, n_pos_angles * 2, 2) * -(np.log(base) / (n_pos_angles * 2)))

        pos_angles = position * div_term  # (max_len, n_pos_angles)
        pos_angles = np.expand_dims(pos_angles, 0)  # (1, max_len, n_pos_angles)
        self.pos_angles = tf.constant(pos_angles, dtype=tf.float32)

        # --- 2. Precompute Cyclical Rates ---
        # Store rates (2pi / period) for each feature to compute angles dynamically
        self.cyclical_rates = []

        # We maintain a sorted list of keys to ensure the order of `tt` matches the rates
        self.sorted_features = sorted(self.time_features.keys())

        if self.time_features:
            for feature in self.sorted_features:
                period = self.time_features[feature]
                # rate = 2*pi / period
                rate = 2 * np.pi / period
                self.cyclical_rates.append(rate)

            # Convert to tensor for efficient multiplication: shape (1, 1, n_time_features)
            self.cyclical_rates_tensor = tf.constant(self.cyclical_rates, dtype=tf.float32)
            self.cyclical_rates_tensor = tf.reshape(self.cyclical_rates_tensor, (1, 1, -1))

    def call(self, tt):
        # tt: (Batch, Seq, n_features)
        # Note: tt is expected to contain the integer time values corresponding to self.sorted_features

        # 1. Get Standard Positional Angles
        # Slice to current sequence length
        seq_len = tf.shape(tt)[1]
        p_angles = self.pos_angles[:, :seq_len, :]
        p_angles = tf.tile(p_angles, [tf.shape(tt)[0], 1, 1])  # (Batch, Seq, n_pos_angles)

        # 2. Get Cyclical Angles
        if self.time_features:
            # tt contains values for the time features.
            # Multiplies (Batch, Seq, n_feats) * (1, 1, n_feats) -> (Batch, Seq, n_feats)
            # We cast tt to float32 to perform the multiplication
            c_angles = tf.cast(tt, tf.float32) * self.cyclical_rates_tensor

            # 3. Concatenate: [Positional Angles, Cyclical Angles]
            return tf.concat([p_angles, c_angles], axis=-1)

        return p_angles
    

def first_quadrant_sincos_pairs(length):
    positions = np.arange(length)[:, np.newaxis]  # (length, 1)
    angles = (positions / length) * (np.pi / 2)  # Scale to [0, pi/2]
    encoding = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # (length, 2)
    return encoding


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, pos_enc=True, time_features=None, activation="relu", l2_reg=None, custom_embedding=None):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.pos_enc = pos_enc
        self.time_features = [] if time_features is None else time_features
        self.activation = activation
        self.custom_embedding = custom_embedding

        self.l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        if self.activation == 'swiglu':
            from ists_.model.utils import SwiGLUConv1D
            self.embedding = SwiGLUConv1D(self.d_model, self.kernel_size, padding='same', l2_reg=l2_reg)
        else:
            self.embedding = tf.keras.layers.Conv1D(
                filters=self.d_model, kernel_size=self.kernel_size, padding='same',
                activation=self.activation, kernel_regularizer=self.l2_reg,
            )

        if self.custom_embedding == 1:
            # Custom Strategy: Combined Embeddings
            self.pos_enc = True
            self.pos_embedder = CustomPositionalEmbedding(d_model=d_model, time_features={
                f: TIME_N_VALUES[f] for f in time_features
            }, base=1000)
            '''self.pos_embedder = AnglesEmbedding(d_model=d_model, time_features={
                f: TIME_N_VALUES[f] for f in time_features
            }, base=1000)'''
        elif self.custom_embedding == 2:
            if self.pos_enc:
                self.pos_embedder = PositionalEmbedding(self.d_model, base=1000)
            self.proj = tf.keras.layers.Dense(d_model, kernel_regularizer=self.l2_reg)
            if self.time_features:
                self.time_lookup = TimeEmbedding({f: TIME_N_VALUES[f] for f in time_features})
        elif self.custom_embedding in [3, 6]:
            if self.pos_enc:
                self.pos_embedder = PositionalEmbedding(self.d_model, base=1000)
            if self.time_features:
                # self.time_embedders = [CyclicalEmbedding(c_in=TIME_N_VALUES[f]) for f in time_features]
                self.time_lookup = TimeEmbedding({f: TIME_N_VALUES[f] for f in time_features})
                self.proj = tf.keras.layers.Dense(
                    d_model, kernel_regularizer=self.l2_reg, name="time_embeddings_proj"
                )
        elif self.custom_embedding == 5:
            if self.pos_enc:
                self.pos_embedder = PositionalEmbedding(self.d_model, base=1000)
            if self.time_features:
                self.time_lookup = TimeEmbedding({f: TIME_N_VALUES[f] for f in time_features})
        elif self.custom_embedding in [4, 7]:
            self.pos_enc = False
            self.pe = None
            if self.time_features:
                self.time_lookup = TimeEmbedding({f: TIME_N_VALUES[f] for f in time_features})
                self.proj = tf.keras.layers.Dense(d_model, kernel_regularizer=self.l2_reg)
        else:
            # Legacy Strategy: Separate Embeddings
            if self.pos_enc:
                self.pos_embedder = PositionalEmbedding(self.d_model, base=1000)
            if self.time_features:
                self.time_embedders = [FixedEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f], base=1000) for f in time_features]
                # self.time_embedders = [TrainablePeriodicEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f]) for f in time_features]

        self.ve_scale = np.sqrt(self.d_model/2)

    def build(self, x_shape, tt_shape):
        _, C, T, _ = x_shape

        if self.custom_embedding in [1, 3, 6, None]:
            variable_embeddings = centered_unit_simplex_embeddings(C, self.d_model)  # (V, d_model)
            variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)
            ve_trainable = True
        elif self.custom_embedding == 2:
            variable_embeddings = centered_unit_simplex(C)
            variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, V)
            ve_trainable = True
            # self.ve_proj = tf.keras.layers.Dense(self.d_model, kernel_regularizer=self.l2_reg, name='ve_proj')  # , activation=self.activation)
        elif self.custom_embedding in [4, 7]:
            variable_embeddings = centered_unit_simplex_embeddings(C, self.d_model)  # (V, d_model)
            variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, d_model)
            ve_trainable = True
            with tf.init_scope():
                self.pe = tf.constant(first_quadrant_sincos_pairs(T), dtype=tf.float32)  # (T, 2)
        elif self.custom_embedding == 5:
            D = self.d_model - 2 * len(self.time_features)
            variable_embeddings = centered_unit_simplex_embeddings(C, D)  # (V, D)
            variable_embeddings = variable_embeddings[tf.newaxis, :, tf.newaxis, :]  # (1, V, 1, D)
            ve_trainable = True
            self.ve_scale = np.sqrt(D/2)

        self.variable_embeddings = self.add_weight(
            shape=variable_embeddings.shape,
            initializer=tf.constant_initializer(variable_embeddings),
            trainable=ve_trainable,
            name='variable_embeddings'
        )

        if tt_shape[-1] != len(self.time_features):
            raise ValueError(f'The number of time features provided ({tt_shape[-1]}) does not match the expected ({len(self.time_features)})')

    def call(self, x, tt, **kwargs):  # x: (B, C, T, I), tt: (B, C, T, F)
        x_shape = tf.shape(x)
        B, C, T, I = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        # Embedding values
        x = tf.reshape(x, (B*C, T, I))  # (B*C, T, I)

        emb = self.embedding(x)  # (B*C, T, d_model)

        D = tf.shape(emb)[-1]
        emb = tf.reshape(emb, (B, C, T, D))  # (B, C, T, d_model)

        if self.custom_embedding == 1:
            # This factor sets the relative scale of the embedding and positional_encoding.
            emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            # emb_rms = tf.sqrt(tf.reduce_mean(tf.square(emb), axis=-1, keepdims=True) + 1e-6)
            # emb = emb / emb_rms  # norm=sqrt(d_model)
            emb *= tf.math.sqrt(tf.cast(1 + 1, tf.float32))  # fixme: position + variable

            pos_emb = self.pos_embedder(tt)  # (B, C, T, d_model)  ##1
            '''angles = self.pos_embedder(tt)  ##2
            pos_emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)  ##2'''
            emb = emb + pos_emb

            variable_embeddings = self.variable_embeddings
            variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
            variable_embeddings = variable_embeddings * self.ve_scale  # (1, V, 1, e) -> (1, V, 1, e)
            emb = emb + variable_embeddings

            return emb

        if self.custom_embedding == 2:
            emb = [emb]
            if self.time_features:
                time_emb = self.time_lookup(tt)  # (B, C, T, F*2)
                emb.append(time_emb)

            ch_emb = self.variable_embeddings  # (1, V, 1, V)
            ch_emb = ch_emb / tf.norm(ch_emb, axis=-1, keepdims=True)
            ch_emb = tf.broadcast_to(ch_emb, [B, C, T, C])  # (B, V, T, V)
            emb.append(ch_emb)

            emb = tf.concat(emb, axis=-1)
            emb = self.proj(emb)  # (B, C, T, d_model)

            if self.pos_enc:
                pos_emb = self.pos_embedder(x)  # (1, T, d_model)
                emb = emb + pos_emb[tf.newaxis]

            return emb
        
        if self.custom_embedding in [3, 6]:
            # This factor sets the relative scale of the embedding and positional_encoding.
            if self.custom_embedding == 3:
                emb *= np.sqrt(self.d_model)
            else:  # custom_embedding == 6
                emb *= tf.math.sqrt(tf.cast(1 + 1 + (1 if self.pos_enc else 0), tf.float32))  # time + variable + position

            # time_emb = []
            # for i, time_embedder in enumerate(self.time_embedders):
            #     t = time_embedder(tf.gather(tt, i, axis=-1))
            #     time_emb.append(t)
            # time_emb = tf.concat(time_emb, axis=-1)  # (B, T, F*2)

            time_emb = self.time_lookup(tt)  # (B, T, F*2)
            time_emb = self.proj(time_emb)  # (B, T, d_model)
            time_emb = time_emb / tf.norm(time_emb, axis=-1, keepdims=True) * self.ve_scale
            time_emb = time_emb[:, tf.newaxis]  # (B, 1, T, d_model)
            emb = emb + time_emb * ((1/6) ** 0.5)

            variable_embeddings = self.variable_embeddings  # (1, V, 1, d_model)
            # variable_embeddings = self.ve_proj(self.variable_embeddings)  # (1, V, 1, d_model)
            variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
            variable_embeddings = variable_embeddings * self.ve_scale
            emb = emb + variable_embeddings * ((1/6) ** 0.5)

            if self.pos_enc:
                pos_emb = self.pos_embedder(x)  # (1, T, d_model)
                emb = emb + tf.cast(pos_emb[tf.newaxis], tf.bfloat16) * ((1/6) ** 0.5)

            return emb

        if self.custom_embedding in [4, 7]:
            # This factor sets the relative scale of the embedding and positional_encoding.
            if self.custom_embedding == 4:
                emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            else:  # custom_embedding == 7
                emb *= tf.math.sqrt(tf.cast(1 + (1 if self.pos_enc else 0), tf.float32))  # fixme: time + variable

            time_emb = self.time_lookup(tt)  # (B, C, T, F*2)
            pe = self.pe[tf.newaxis, tf.newaxis, :, :]  # (1, 1, T, 2)
            pe = tf.broadcast_to(pe, [B, C, T, 2])  # (B, C, T, 2)
            time_emb = tf.concat([time_emb, pe], axis=-1)  # (B, C, T, F*2 + 2)
            time_emb = self.proj(time_emb)  # (B, C, T, d_model)
            time_emb = time_emb / tf.norm(time_emb, axis=-1, keepdims=True) * tf.math.sqrt(tf.cast(self.d_model/2, tf.float32))
            emb = emb + time_emb

            variable_embeddings = self.variable_embeddings  # (1, V, 1, d_model)
            variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
            variable_embeddings = variable_embeddings * self.ve_scale
            emb = emb + variable_embeddings

            return emb

        if self.custom_embedding == 5:
            # This factor sets the relative scale of the embedding and positional_encoding.
            emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            emb *= tf.math.sqrt(tf.cast(1 + (1 if self.pos_enc else 0), tf.float32))  # metadata (time features and channel) + position if self.pos_ens=True

            time_emb = self.time_lookup(tt)  # (B, C, T, F*2)
            chan_emb = self.variable_embeddings  # (1, V, 1, D)
            chan_emb = chan_emb / tf.norm(chan_emb, axis=-1, keepdims=True) * self.ve_scale
            chan_emb = tf.broadcast_to(chan_emb, [B, C, T, tf.shape(chan_emb)[-1]])  # (B, V, T, D)
            meta_emb = tf.concat([time_emb, chan_emb], axis=-1)  # metadata: (B, C, T, d_model)
            emb = emb + meta_emb

            if self.pos_enc:
                pos_emb = self.pos_embedder(x)  # (1, T, d_model)
                emb = emb + pos_emb[tf.newaxis]

            return emb

        # --- Legacy Implementation ---
        # This factor sets the relative scale of the embedding and positional_encoding.
        emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb *= tf.math.sqrt(tf.cast(len(self.time_features) + (1 if self.pos_enc else 0) + 1, tf.float32))  # fixme: time features + position + variable

        if self.pos_enc:
            pos_emb = self.pos_embedder(x)
            emb = emb + pos_emb

        if self.time_features:
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(tf.gather(tt, i, axis=-1))
                emb = emb + time_emb

        variable_embeddings = self.variable_embeddings  # (1, V, 1, d_model)
        # variable_embeddings = self.ve_proj(self.variable_embeddings)  # (1, V, 1, d_model)
        variable_embeddings /= tf.norm(variable_embeddings, axis=-1, keepdims=True)  # normalize to unit length
        variable_embeddings = variable_embeddings * self.ve_scale
        emb = emb + variable_embeddings

        return emb


def centered_unit_simplex(N: int):
    basis_vectors = np.eye(N)

    centroid = np.mean(basis_vectors, axis=0, keepdims=True)
    centered_vectors = basis_vectors - centroid

    scaling_factor = np.sqrt(float(N) / float(N - 1))
    scaled_vectors = centered_vectors * scaling_factor

    return scaled_vectors


def centered_unit_simplex_embeddings(N: int, embedding_dim: int):
    scaled_vectors = centered_unit_simplex(N)

    padding_dims = max(0, embedding_dim - N)

    paddings = ((0, 0), (0, padding_dims))
    padded_embeddings = np.pad(
        scaled_vectors, paddings, "constant", constant_values=0
    )

    # Create a random square matrix of shape (E, E)
    random_matrix = np.random.randn(embedding_dim, embedding_dim)

    # Use QR decomposition to get an orthogonal matrix Q
    q_matrix, _ = np.linalg.qr(random_matrix)

    # Apply the rotation to the padded embeddings
    dense_embeddings = np.dot(padded_embeddings, q_matrix)

    return dense_embeddings
