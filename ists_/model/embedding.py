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


'''class FixedEmbedding(tf.keras.layers.Layer):
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
        return self.emb(x)'''


'''def periodic_sinusoidal_encoding(length, depth):
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


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        # w = periodic_sinusoidal_encoding(c_in, d_model)
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


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, time_features=None, activation="relu", l2_reg=None, custom_embedding=False):
        super().__init__()
        self.d_model = d_model
        self.custom_embedding = custom_embedding
        self.time_features = [] if time_features is None else time_features

        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation=activation,
            kernel_regularizer=l2_reg
        )

        if self.custom_embedding:
            ...  # fixme: to be implemented
        else:
            # Legacy Strategy: Separate Embeddings
            # self.pos_embedder = PositionalEmbedding(self.d_model, base=100)
            self.pos_embedder = PositionalEmbedding(self.d_model - 2 * len(self.time_features), base=1000)
            if self.time_features:
                self.time_embedders = [FixedEmbedding(d_model=d_model, c_in=TIME_N_VALUES[f]) for f in time_features]  # base=100

    def build(self, x_shape, tt_shape):
        if tt_shape[-1] != len(self.time_features):
            raise ValueError(f'The number of time features provided ({tt_shape[-1]}) does not match the expected ({len(self.time_features)})')

    def call(self, x, tt, **kwargs):  # x: (B, T, C), tt: (B, T, F)

        # Embedding values
        emb = self.embedding(x)

        # This factor sets the relative scale of the embedding and positional_encoding.
        emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        if self.custom_embedding:
            return None  # fixme: to be implemented
        # --- Legacy Implementation ---
        # emb *= tf.math.sqrt(tf.cast(len(self.time_features) + 1 + 1, tf.float32))  # fixme: time features + position + variable
        emb *= tf.math.sqrt(tf.cast(1 + 1, tf.float32))  # fixme: position + variable

        pos_emb = self.pos_embedder(x)
        pos_emb = [tf.tile(pos_emb, [tf.shape(x)[0], 1, 1])]

        if self.time_features:
            arr_times = tt
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(tf.gather(arr_times, i, axis=-1))
                # emb = emb + time_emb
                pos_emb.append(time_emb)
        pos_emb = tf.concat(pos_emb, axis=-1)

        emb = emb + pos_emb

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
