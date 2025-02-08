import numpy as np
import tensorflow as tf

from ists.preprocessing import TIME_N_VALUES


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


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, feature_mask, is_null_embedding=False, time_features=None):
        super().__init__()
        # Embedding dimension & layer
        self.d_model = d_model
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation='gelu'
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

        # Null positional embedding layer
        self.null_embedder = None
        if is_null_embedding:
            self.null_embedder = FixedEmbedding(d_model=d_model, c_in=2)

        self.feat_ids = [i for i, x in enumerate(feature_mask) if x == 0]
        self.null_id = [i for i, x in enumerate(feature_mask) if x == 1]
        self.time_ids = [i for i, x in enumerate(feature_mask) if x == 2]

    def call(self, x, **kwargs):
        if tf.shape(x)[2] != len(self.feature_mask):
            raise ValueError(f'Input data {tf.shape(x)} have a different features dimension that the provided feature mask ({len(self.feature_mask)})')

        # Extract value, null, and time array from the input matrix
        values = tf.gather(x, self.feat_ids, axis=2)

        # Embedding values
        value_emb = self.embedding(values)

        # This factor sets the relative scale of the embedding and positional_encoding.
        value_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        emb = value_emb + self.pos_embedder(x)

        # Add the time encoding
        if self.time_embedders:
            arr_times = tf.gather(x, self.time_ids, axis=2)
            for i, time_embedder in enumerate(self.time_embedders):
                time_emb = time_embedder(arr_times[:, :, i])
                emb = emb + time_emb

        # Add the null encoding
        if self.null_embedder:
            null_emb = self.null_embedder(x[:, :, self.null_id[0]])
            emb = emb + null_emb

        return emb
