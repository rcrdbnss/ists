import numpy as np
import tensorflow as tf

from ists_.preprocessing import TIME_N_VALUES




class PhysicalSinusoidalEmbedding(tf.keras.layers.Layer):
    """
    Physically grounded sinusoidal embedding.
    - Cyclical Features (Day, Month): Scaled by 2pi (Perfect wrapping).
    - Linear Features (Position): Scaled by 1.0 (Monotonic, non-wrapping).
    - Initialization: Log-Linear P (Geometric) based on Capacity K.
    - Weights: W = S * exp(P)
    """
    def __init__(self, d_model, feature_configs):
        """
        feature_configs: List of dicts defining each feature.
        Example:
        [
            {'name': 'Pos', 'K': 1000, 'type': 'linear'},     # Scaling = 1.0
            {'name': 'Day', 'K': 31,   'type': 'cyclical'},   # Scaling = 2*pi
            {'name': 'Month', 'K': 12,   'type': 'cyclical'}  # Scaling = 2*pi
        ]
        """
        super(PhysicalSinusoidalEmbedding, self).__init__()
        self.d_model = d_model
        self.d_half = d_model // 2

        n_features = len(feature_configs)

        # Arrays to hold initialization and scaling factors
        init_P = np.zeros((n_features, self.d_half), dtype=np.float32)
        scaling_factors = np.zeros((n_features, 1), dtype=np.float32)
        
        for i, config in enumerate(feature_configs):
            K = config['K']
            feat_type = config.get('type', 'linear')
            
            # --- 1. Determine Scaling Factor ---
            if feat_type == 'linear':
                # Position: Max arg = 1.0 radian (Monotonic)
                s = 1.0
            else:
                # Time: Max arg = 2pi radians (Cyclic)
                s = 2 * np.pi
                
            scaling_factors[i] = s
            
            # --- 2. Initialize P (Log Frequencies) ---
            # We want the lowest freq to complete 1 "unit" over K
            # Linear: 1 wave = length K (freq = 1/K)
            # Cyclic: 1 wave = length K (freq = 1/K)
            # The scaling factor 's' handles the 2pi conversion later.
            
            # Bounds for Integers:
            # Min Freq: 1/K
            # Max Freq: 1.0
            log_min = np.log(1.0 / K)
            log_max = np.log(1.0) 
            
            # Generate random log-linear frequencies for this specific feature
            # We sample d_half frequencies
            row_P = np.random.uniform(log_min, log_max, size=(self.d_half,))
            init_P[i, :] = row_P

        # Create the Learnable Weights
        self.log_frequencies = self.add_weight(
            shape=(n_features, self.d_half),
            initializer=tf.constant_initializer(init_P),
            trainable=True,
            name='log_frequencies_P'
        )

        # Scaling is FIXED (non-trainable structural constant)
        # Shape (n_features, 1) for broadcasting
        self.scaling_factors = tf.constant(scaling_factors, dtype=tf.float32)

    def call(self, x):
        # x shape: (batch, seq, n_features) - RAW INTEGERS
        x = tf.cast(x, tf.float32)
        
        # 1. Recover W = exp(P)
        W = tf.math.exp(self.log_frequencies)
        
        # 2. Apply Hybrid Scaling
        # We multiply W by the scaling factor specific to each feature row
        # W shape: (n_features, d_half)
        # S shape: (n_features, 1)
        W_scaled = W * self.scaling_factors
        
        # 3. Projection
        projection = tf.tensordot(x, W_scaled, axes=[[-1], [0]])
        
        # 4. Activation (Sin + Cos pairs)
        return tf.concat([tf.math.sin(projection), tf.math.cos(projection)], axis=-1)


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


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, feature_mask, time_features=None, activation="relu", l2_reg=None, mixed_strategy=False):
        super().__init__()
        self.d_model = d_model
        self.mixed_strategy = mixed_strategy  # New flag for Strategy B
        self.time_features = time_features

        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
            kernel_regularizer=l2_reg
        )

        # Feature mask to split values for time encodings and null encoding
        self.feature_mask = np.array(feature_mask)
        if time_features and len(self.feature_mask[self.feature_mask == 2]) != len(time_features):
            raise ValueError('time_features must have the same dimension of the number of time features')

        self.time_embedders = []
        self.mixed_embedder = None

        if self.mixed_strategy:
            # Strategy B: 1 Position dim + N time feature dims
            self.pos_base = 100
            self.mixed_embedder = PhysicalSinusoidalEmbedding(d_model, feature_configs=(
                [{'name': 'Pos', 'K': self.pos_base, 'type': 'linear'}] + 
                ([{'name': f, 'K': TIME_N_VALUES[f], 'type': 'cyclical'} for f in time_features] if time_features else [])
            ))
        else:
            # Legacy Strategy: Separate Embeddings
            self.pos_embedder = PositionalEmbedding(self.d_model, base=100)
            if time_features:
                self.time_embedders = [FixedEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f], base=100) for f in time_features]

        self.feat_ids = [i for i, x in enumerate(feature_mask) if x == 0]
        self.time_ids = [i for i, x in enumerate(feature_mask) if x == 2]

    def call(self, x, **kwargs):
        # Extract value, null, and time array from the input matrix
        values = tf.gather(x, self.feat_ids, axis=-1)

        # Embedding values
        emb = self.embedding(values)

        # This factor sets the relative scale of the embedding and positional_encoding.
        emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        if self.mixed_strategy:
            # --- STRATEGY B Implementation ---
            # 1. Get raw time features and cast to float
            arr_times = tf.cast(tf.gather(x, self.time_ids, axis=-1), tf.float32)
            
            # 2. Normalize time features (0 to 1) using known max values
            # We iterate to divide each feature by its specific max value
            norm_times_list = []
            for i, feat_name in enumerate(self.time_features):
                max_val = float(TIME_N_VALUES[feat_name])
                # Slice, normalize, and keep dims
                norm_times_list.append(arr_times[:, :, i:i+1] / max_val)
            norm_times = tf.concat(norm_times_list, axis=-1) if norm_times_list else arr_times

            # 3. Create Position sequence (0..L), cast and normalize
            seq_len = tf.shape(x)[1]
            pos_seq = tf.range(seq_len, dtype=tf.float32)
            pos_seq = tf.expand_dims(pos_seq, 0) # Batch dim
            pos_seq = tf.expand_dims(pos_seq, -1) # Feature dim
            # Broadcast to batch size
            pos_seq = tf.broadcast_to(pos_seq, [tf.shape(x)[0], seq_len, 1])
            # Normalize position
            norm_pos = pos_seq / self.pos_base
            # norm_pos = pos_seq / tf.cast(seq_len, tf.float32)

            # 4. Concatenate [Position, Time1, Time2...]
            # mixed_input = tf.concat([pos_seq, arr_times], axis=-1)  # raw values
            mixed_input = tf.concat([norm_pos, norm_times], axis=-1)

            # 5. Apply Mixed Sinusoidal Layer
            # We add this to the value embedding (similar to how pos encoding is usually added)
            emb *= tf.math.sqrt(1.0 + 1.0)  # fixme: time/pos + variable
            emb = emb + self.mixed_embedder(mixed_input)

            return emb

        # --- Legacy Implementation ---
        emb *= tf.math.sqrt(tf.cast(len(self.time_embedders) + 1 + 1, tf.float32))  # fixme: time features + position + variable
        emb = emb + self.pos_embedder(x)

        if self.time_embedders:
            arr_times = tf.gather(x, self.time_ids, axis=-1)
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(tf.gather(arr_times, i, axis=-1))
                emb = emb + time_emb

        return emb


def variable_embeddings_regular_simplex(num_variables: int, embedding_dim: int):
    basis_vectors = np.eye(num_variables)

    centroid = np.mean(basis_vectors, axis=0, keepdims=True)
    centered_vectors = basis_vectors - centroid

    scaling_factor = np.sqrt(float(num_variables) / float(num_variables - 1))  # * np.sqrt(embedding_dim / 2)
    scaled_vectors = centered_vectors * scaling_factor

    padding_dims = max(0, embedding_dim - num_variables)

    paddings = ((0, 0), (0, padding_dims))
    final_embeddings = np.pad(
        scaled_vectors, paddings, "constant", constant_values=0
    )

    return final_embeddings


def variable_embeddings_regular_simplex_dense(num_variables: int, embedding_dim: int):
    padded_embeddings = variable_embeddings_regular_simplex(num_variables, embedding_dim)

    # Create a random square matrix of shape (E, E)
    random_matrix = np.random.randn(embedding_dim, embedding_dim)

    # Use QR decomposition to get an orthogonal matrix Q
    q_matrix, _ = np.linalg.qr(random_matrix)

    # Apply the rotation to the padded embeddings
    dense_embeddings = np.dot(padded_embeddings, q_matrix)

    return dense_embeddings
