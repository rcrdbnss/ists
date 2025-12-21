import numpy as np
import tensorflow as tf

from ists_.preprocessing import TIME_N_VALUES


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000, base=10000.0):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(base) / d_model))

        # pe = np.zeros((max_len, d_model), dtype=np.float32)
        # pe[:, 0::2] = np.sin(position * div_term)
        # pe[:, 1::2] = np.cos(position * div_term)
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

        # w = np.zeros((c_in, d_model), dtype=np.float32)
        # w[:, 0::2] = np.sin(position * div_term)
        # w[:, 1::2] = np.cos(position * div_term)
        w = np.concatenate([np.sin(position * div_term), np.cos(position * div_term)], axis=1)

        # Initialize the embedding layer with the precomputed weights
        self.emb = tf.keras.layers.Embedding(c_in, d_model, embeddings_initializer=tf.constant_initializer(w),
                                             trainable=False)

    def call(self, x):
        return self.emb(x)


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

        self.pos_embedder = PositionalEmbedding(self.d_model, base=1000)

        # Feature mask to split values for time encodings and null encoding
        self.feature_mask = np.array(feature_mask)
        if is_null_embedding and 1 not in feature_mask:
            raise ValueError('Null embedding is set to True but no null feature is provided in the feature mask')
        if time_features and len(self.feature_mask[self.feature_mask == 2]) != len(time_features):
            raise ValueError('time_features must have the same dimension of the number of time features')

        # Time embedding layers
        self.time_embedders = []
        if time_features:
            self.time_embedders = [FixedEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f], base=1000) for f in time_features]
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
        emb *= tf.math.sqrt(tf.cast(2 + 1 + 1, tf.float32))  # fixme: time features + position + variable

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
