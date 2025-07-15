import numpy as np
import tensorflow as tf

from ists_.preprocessing import TIME_N_VALUES


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        pe = np.expand_dims(pe, 0)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return self.pe[:, :tf.shape(x)[1], :]


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # Create the embedding matrix
        w = np.zeros((c_in, d_model), dtype=np.float32)

        position = np.expand_dims(np.arange(0, c_in), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        w[:, 0::2] = np.sin(position * div_term)
        w[:, 1::2] = np.cos(position * div_term)

        # Initialize the embedding layer with the precomputed weights
        self.emb = tf.keras.layers.Embedding(c_in, d_model, embeddings_initializer=tf.constant_initializer(w),
                                             trainable=False)

    def call(self, x):
        return self.emb(x)


"""class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super().__init__()

        # Create the embedding matrix
        w = np.zeros((c_in, d_model), dtype=np.float32)

        # 1. Calculate the base angle for each position in the cycle
        position = np.expand_dims(np.arange(0, c_in), 1)
        angles = 2 * np.pi * position / c_in

        # 2. Define the integer frequencies for the Fourier features
        freqs = np.expand_dims(np.arange(1, d_model // 2 + 1), 0)
        # freqs = np.ones((1, d_model // 2))

        # 3. Calculate the sine and cosine values using broadcasting
        w[:, 0::2] = np.sin(angles * freqs)
        w[:, 1::2] = np.cos(angles * freqs)

        # Initialize the embedding layer with the precomputed weights
        self.emb = tf.keras.layers.Embedding(
            c_in, d_model, embeddings_initializer=tf.constant_initializer(w),
            trainable=True,
            # trainable=False,
        )

    def call(self, x):
        return self.emb(x)"""


"""class CyclicalEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super(CyclicalEmbedding, self).__init__()
        self.c_in = c_in
        self.proj = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x):
        theta = 2 * np.pi * x / self.c_in  # (B, T)
        theta = tf.expand_dims(theta, axis=-1)  # (B, T, 1)
        sin_emb = tf.sin(theta)
        cos_emb = tf.cos(theta)
        emb = tf.concat([sin_emb, cos_emb], axis=-1)  # (B, T, 2)
        emb = self.proj(emb)  # (B, T, d_model)
        return emb"""


class CyclicalEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super(CyclicalEmbedding, self).__init__()

        # 1. Start with the 2D representation
        position = np.expand_dims(np.arange(0, c_in), 1)  # (C, 1)
        angles = 2 * np.pi * position / c_in
        embeddings_2d = np.c_[np.sin(angles), np.cos(angles)]  # (C, 2)

        """# 2. Pad with zeros to reach d_model dimensions
        num_samples = len(position)
        embeddings = np.zeros((num_samples, d_model))  # (C, d_model)
        embeddings[:, :2] = embeddings_2d"""
        """# 2. Repeat the 2D embeddings to fill d_model dimensions
        repeat_factor = d_model // 2
        embeddings = np.tile(embeddings_2d, (1, repeat_factor))  # (C, d_model)"""

        """# 3. Create a random orthogonal matrix Q using QR decomposition
        random_matrix = np.random.randn(d_model, d_model)
        q, _ = np.linalg.qr(random_matrix)

        # 4. Multiply the padded vectors by Q
        embeddings = embeddings @ q"""

        # 2. Project to d_model dimensions
        random_matrix = np.random.normal(loc=0.0, scale=1.0, size=(2, d_model))
        embeddings = np.dot(embeddings_2d, random_matrix)  # (C, d_model)

        # 3. Normalize to unit hypersphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        self.emb = tf.keras.layers.Embedding(
            c_in, d_model, embeddings_initializer=tf.constant_initializer(embeddings),
            # trainable=True,
            trainable=False,
        )

        # self.scale = 1.0
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer='ones',
            trainable=True,
        )

        # self.bias = 0.0
        self.bias = self.add_weight(
            name='bias',
            shape=(),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        return self.emb(x) * self.scale + self.bias


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, feature_mask, is_null_embedding=False, time_features=None, activation="relu", l2_reg=None):
        super().__init__()
        # Embedding dimension & layer
        self.d_model = d_model
        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
            kernel_regularizer=l2_reg
        )

        self.pos_embedder = PositionalEmbedding(self.d_model)

        # Feature mask to split values for time encodings and null encoding
        self.feature_mask = np.array(feature_mask)
        if is_null_embedding and 1 not in feature_mask:
            raise ValueError('Null embedding is set to True but no null feature is provided in the feature mask')
        if time_features and len(self.feature_mask[self.feature_mask == 2]) != len(time_features):
            raise ValueError('time_features must have the same dimension of the number of time features')

        # Time embedding layers
        self.time_embedders = []
        if time_features:
            self.time_embedders = [FixedEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f]) for f in time_features]
            # self.time_embedders = [CyclicalEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f]) for f in time_features]
        self.time_feats_scale = 1.0  # fixed scale
        """# learnable scale factor
        self.time_feats_scale = self.add_weight(
            name='time_feats_scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
            dtype=tf.float32
        )"""

        # Null positional embedding layer
        self.null_embedder = None
        if is_null_embedding:
            self.null_embedder = FixedEmbedding(d_model=d_model, c_in=2)

        self.feat_ids = [i for i, x in enumerate(feature_mask) if x == 0]
        self.null_id = [i for i, x in enumerate(feature_mask) if x == 1]
        if self.null_id: # If null_id is not empty, we take the first one
            self.null_id = self.null_id[0]
        self.time_ids = [i for i, x in enumerate(feature_mask) if x == 2]

    def call(self, x, **kwargs):
        # if tf.shape(x)[2] != len(self.feature_mask):
        #     raise ValueError(f'Input data {tf.shape(x)} have a different features dimension that the provided feature mask ({len(self.feature_mask)})')

        # Extract value, null, and time array from the input matrix
        values = tf.gather(x, self.feat_ids, axis=-1)

        # Embedding values
        emb = self.embedding(values)

        # This factor sets the relative scale of the embedding and positional_encoding.
        emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        emb = emb + self.pos_embedder(x)

        # Add the time encoding
        if self.time_embedders:
            arr_times = tf.gather(x, self.time_ids, axis=-1)
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(tf.gather(arr_times, i, axis=-1))
                emb = emb + self.time_feats_scale * time_emb

        # Add the null encoding
        if self.null_embedder:
            null_emb = self.null_embedder(x[:, :, self.null_id])
            emb = emb + null_emb

        return emb


def create_fixed_variable_embeddings(num_variables: int, embedding_dim: int):
    basis_vectors = tf.eye(num_variables)

    centroid = tf.reduce_mean(basis_vectors, axis=0, keepdims=True)
    centered_vectors = basis_vectors - centroid

    scaling_factor = tf.sqrt(float(num_variables) / float(num_variables - 1))
    scaled_vectors = centered_vectors * scaling_factor

    padding_dims = embedding_dim - num_variables
    if padding_dims < 0:
        padding_dims = 0

    paddings = tf.constant([[0, 0], [0, padding_dims]])
    final_embeddings = tf.pad(
        scaled_vectors, paddings, "CONSTANT", constant_values=0
    )

    return final_embeddings


def create_fixed_dense_variable_embeddings(num_variables: int, embedding_dim: int):
    padded_embeddings = create_fixed_variable_embeddings(num_variables, embedding_dim)

    # Create a random square matrix of shape (E, E)
    random_matrix = tf.random.normal(shape=(embedding_dim, embedding_dim))

    # Use QR decomposition to get an orthogonal matrix Q
    q_matrix, _ = tf.linalg.qr(random_matrix)

    # Apply the rotation to the padded embeddings
    dense_embeddings = tf.matmul(padded_embeddings, q_matrix)

    return dense_embeddings
