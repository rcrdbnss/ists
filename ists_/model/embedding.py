import numpy as np
import tensorflow as tf

from ists_.preprocessing import TIME_N_VALUES


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, base=10000.0):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))

        pe = np.concatenate([np.sin(position * div_term), np.cos(position * div_term)], axis=1)

        pe = np.expand_dims(pe, 0)
        self.pe = tf.constant(pe, dtype=tf.float32)

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


class CustomPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, time_features=None, max_len=5000, base=10000.0):
        super().__init__()
        self.pos_embedder = PositionalEmbedding(d_model - 2 * len(time_features), max_len=max_len, base=base)
        self.time_features = {} if time_features is None else time_features  # {name: c_in}
        if self.time_features:
            self.time_embedders = [CyclicalEmbedding(c_in=c_in) for c_in in self.time_features.values()]

    def call(self, tt):
        pos_emb = self.pos_embedder(tt)
        pos_emb = [tf.tile(pos_emb, [tf.shape(tt)[0], 1, 1])]

        if self.time_features:
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(tf.gather(tt, i, axis=-1))
                pos_emb.append(time_emb)
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


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, pos_enc=True, time_features=None, activation="relu", l2_reg=None, custom_embedding=None):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.time_features = [] if time_features is None else time_features
        self.custom_embedding = custom_embedding

        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
            kernel_regularizer=l2_reg
        )

        if self.custom_embedding == 1:
            # Custom Strategy: Combined Embeddings
            '''self.pos_embedder = CustomPositionalEmbedding(d_model=d_model, time_features={
                f: TIME_N_VALUES[f] for f in time_features
            }, base=1000)'''
            self.pos_enc = True
            self.pos_embedder = AnglesEmbedding(d_model=d_model, time_features={
                f: TIME_N_VALUES[f] for f in time_features
            }, base=1000)
        elif self.custom_embedding in [2, 3]:
            self.pos_enc = False
            if self.time_features:
                self.time_embedders = [CyclicalEmbedding(c_in=TIME_N_VALUES[f]) for f in time_features]
                self.proj = tf.keras.layers.Dense(d_model, kernel_regularizer=l2_reg)
        else:
            # Legacy Strategy: Separate Embeddings
            if self.pos_enc:
                self.pos_embedder = PositionalEmbedding(self.d_model, base=1000)
            if self.time_features:
                self.time_embedders = [FixedEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f], base=1000) for f in time_features]
                # self.time_embedders = [TrainablePeriodicEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f]) for f in time_features]

    def build(self, x_shape, tt_shape):
        if tt_shape[-1] != len(self.time_features):
            raise ValueError(f'The number of time features provided ({tt_shape[-1]}) does not match the expected ({len(self.time_features)})')

    def call(self, x, tt, **kwargs):  # x: (B, T, C), tt: (B, T, F)
        # Embedding values
        emb = self.embedding(x)

        if self.custom_embedding == 1:
            # This factor sets the relative scale of the embedding and positional_encoding.
            emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            emb *= tf.math.sqrt(tf.cast(1 + 1, tf.float32))  # fixme: position + variable

            # pos_emb = self.pos_embedder(tt)
            angles = self.pos_embedder(tt)
            pos_emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

            emb = emb + pos_emb

            return emb

        if self.custom_embedding == 2:
            time_emb = []
            for i, time_embedder in enumerate(self.time_embedders):
                t = time_embedder(tf.gather(tt, i, axis=-1))
                time_emb.append(t)
            time_emb = tf.concat(time_emb, axis=-1)  # (B, T, F*2)
            emb = tf.concat([emb, time_emb], axis=-1)  # (B, T, d_model + F*2)
            emb = self.proj(emb)  # (B, T, d_model)

            # This factor sets the relative scale of the embedding and positional_encoding.
            emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            return emb
        
        if self.custom_embedding == 3:
            # This factor sets the relative scale of the embedding and positional_encoding.
            # emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  ##1
            emb = emb / (tf.sqrt(tf.reduce_mean(tf.square(emb), axis=-1, keepdims=True) + 1e-6))  # norm=sqrt(d_model)  ##2
            emb *= tf.math.sqrt(tf.cast(1 + 1, tf.float32))  # fixme: position + variable

            time_emb = []
            for i, time_embedder in enumerate(self.time_embedders):
                t = time_embedder(tf.gather(tt, i, axis=-1))
                time_emb.append(t)
            time_emb = tf.concat(time_emb, axis=-1)  # (B, T, F*2)
            time_emb = self.proj(time_emb)  # (B, T, d_model)
            time_emb = time_emb / tf.norm(time_emb, axis=-1, keepdims=True) * tf.math.sqrt(tf.cast(self.d_model//2, tf.float32))

            emb = emb + time_emb
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
