import numpy as np
# import pandas as pd
import tensorflow as tf


def null_encoding(mask, depth):
    pass


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation='gelu'
        )
        # self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, **kwargs):
        # Extract value, null, and time array from the input matrix
        arr_times = x[:, :, -1]
        arr_nulls = x[:, :, -2]
        x = x[:, :, :-2]

        length = tf.shape(x)[1]

        # Embedding values
        x = self.embedding(x)
        # x = self.activation(x)

        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Add the positional encoding
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class SpatialEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, spatial_size):
        super().__init__()
        self.spatial_size = spatial_size
        self.emb_layers = [
            TemporalEmbedding(d_model=d_model, kernel_size=kernel_size)
            for _ in range(spatial_size)
        ]

    def call(self, inputs, **kwargs):
        embedded_inputs = []
        for i in range(self.spatial_size):
            embedded_inputs.append(self.emb_layers[i](inputs[i]))

        embedded_inputs = tf.concat(embedded_inputs, axis=1)

        return embedded_inputs


if __name__ == '__main__':
    # Input
    data = np.random.randn(100, 24, 4).astype(np.float32)

    data1 = np.random.randn(100, 3, 4).astype(np.float32)
    data2 = np.random.randn(100, 3, 4).astype(np.float32)
    data3 = np.random.randn(100, 3, 4).astype(np.float32)
    sdata = [data1, data2, data3]
    # dates = pd.date_range('2020-01-01', periods=24).date
    # data[:, :, -1] = dates

    # PositionalEmbedding init & call
    embedder = TemporalEmbedding(d_model=512, kernel_size=3)
    x_emb = embedder(data)
    print(f'TemporalEmbedding:  {x_emb.shape} {data.shape}')

    # GlobalEmbedding init & call
    global_embedder = SpatialEmbedding(d_model=512, kernel_size=3, spatial_size=len(sdata))
    sx_emb = global_embedder(sdata)
    print(f'GlobalEmbedding:  {sx_emb.shape}')
    print('Hello World!')
