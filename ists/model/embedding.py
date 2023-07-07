import numpy as np
# import pandas as pd
import tensorflow as tf
from datetime import datetime


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return pos_encoding


def tensor_mask_encoding(mask_array, pos_encoding):
    mask_array = tf.cast(mask_array, tf.int32)
    depth = tf.shape(pos_encoding)[1]
    shape = tf.shape(mask_array)
    new_shape = tf.concat([shape, tf.expand_dims(depth, axis=-1)], axis=0)
    mask_encoded = tf.TensorArray(pos_encoding.dtype, size=tf.size(mask_array))

    for i in tf.range(shape[0]):
        if shape.shape[0] > 1:
            for ii in tf.range(shape[1]):
                if shape.shape[0] > 2:
                    for iii in tf.range(shape[2]):
                        mask_encoded = mask_encoded.write(i * shape[1] * shape[2] + ii * shape[2] + iii,
                                                          pos_encoding[mask_array[i, ii, iii]])
                else:
                    mask_encoded = mask_encoded.write(i * shape[1] + ii, pos_encoding[mask_array[i, ii]])
        else:
            mask_encoded = mask_encoded.write(i, pos_encoding[mask_array[i]])

    mask_encoded = mask_encoded.stack()
    mask_encoded = tf.reshape(mask_encoded, new_shape)

    return mask_encoded


def mask_encoding(mask_array, pos_encoding):
    assert mask_array.ndim < 4
    mask_array = mask_array.astype(int)
    depth = pos_encoding.shape[1]
    shape = mask_array.shape
    new_shape = mask_array.shape + tuple([depth])
    mask_encoded = np.zeros(new_shape)

    for i in range(shape[0]):
        if len(shape) > 1:
            for ii in range(shape[1]):
                if len(shape) > 2:
                    for iii in range(shape[2]):
                        mask_encoded[i, ii, iii] = pos_encoding[mask_array[i, ii, iii]]
                else:
                    mask_encoded[i, ii] = pos_encoding[mask_array[i, ii]]
        else:
            mask_encoded[i] = pos_encoding[mask_array[i]]

    return mask_encoded


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_size):
        super().__init__()

        w = tf.cast(positional_encoding(length=max_size, depth=d_model), dtype=tf.float32)

        self.emb = tf.keras.layers.Embedding(max_size, d_model, trainable=False, weights=[w])

    def call(self, x, **kwargs):
        return self.emb(x)


class FixedEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_size):
        super().__init__()
        # Embedding dimension & layer
        self.d_model = d_model
        self.max_size = max_size
        # self.pos_encoding = tf.cast(positional_encoding(length=max_size, depth=d_model), dtype=tf.float32)
        self.embedding = FixedEmbedding(d_model=d_model, max_size=max_size)

    def call(self, x, **kwargs):
        # Check max_size constraint
        # if tf.reduce_max(x) >= self.max_size:
        #     raise ValueError(f'Encounter a value greater than initialized max size: {a} {b} {a >= b}')
        # # Compute the fixed encoding by using the position encoding like a dictionary
        # emb = mask_encoding(x.numpy(), self.pos_encoding.numpy())
        # emb = tf.cast(emb, dtype=tf.float32)

        # Compute the fixed encoding by using the position encoding like a dictionary
        x = tf.cast(x, tf.int32)  # Cast x to integer type
        emb = self.embedding(x)

        return emb


class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, feature_mask, null_max_size=None, time_max_sizes=None):
        super().__init__()
        # Embedding dimension & layer
        self.d_model = d_model
        # self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.embedding = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='same',
            activation='gelu'
        )
        # Feature mask to split values for time encodings and null encoding
        self.feature_mask = np.array(feature_mask)
        if 1 in feature_mask and null_max_size is None:
            raise ValueError('null_max_size must be provided for null encoding step')
        if 2 in feature_mask and time_max_sizes is None:
            raise ValueError('time_max_sizes must be provided for time encoding step')
        if time_max_sizes and len(self.feature_mask[self.feature_mask == 2]) != len(time_max_sizes):
            raise ValueError('time_max_size must have the same dimension of the number of time features')

        # Time positional encoding layers
        self.time_layers = []
        if time_max_sizes:
            self.time_layers = [FixedEncoding(d_model=d_model, max_size=size) for size in time_max_sizes]

        # Null positional encoding layer
        self.null_pos_encoding = None
        if null_max_size:
            self.null_pos_encoding = FixedEncoding(d_model=d_model, max_size=null_max_size)

        self.feat_ids = [i for i, x in enumerate(feature_mask) if x == 0]
        self.null_id = [i for i, x in enumerate(feature_mask) if x == 1]
        self.time_ids = [i for i, x in enumerate(feature_mask) if x == 2]

    def call(self, x, **kwargs):
        if x.shape[2] != len(self.feature_mask):
            raise ValueError('Input data have a different features dimension that the provided feature mask')

        # Extract value, null, and time array from the input matrix
        x_feat = tf.gather(x, self.feat_ids, axis=2)

        # Embedding values
        x_feat = self.embedding(x_feat)
        # x_feat = self.activation(x_feat

        # This factor sets the relative scale of the embedding and positional_encoding.
        x_feat *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Add the time encoding
        if self.time_layers:
            arr_times = tf.gather(x, self.time_ids, axis=2)
            for i, time_layer in enumerate(self.time_layers):
                time_emb = time_layer(arr_times[:, :, i])
                x_feat = x_feat + time_emb

        # Add the null encoding
        if self.null_pos_encoding:
            null_emb = self.null_pos_encoding(x[:, :, self.null_id[0]])
            x_feat = x_feat + null_emb

        return x_feat


class SpatialEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, kernel_size, spatial_size, feature_mask, time_max_sizes=None, null_max_size=None):
        super().__init__()
        self.spatial_size = spatial_size
        self.emb_layers = [
            TemporalEmbedding(
                d_model=d_model,
                kernel_size=kernel_size,
                feature_mask=feature_mask,
                time_max_sizes=time_max_sizes,
                null_max_size=null_max_size
            )
            for _ in range(spatial_size)
        ]

    def call(self, inputs, **kwargs):
        embedded_inputs = []
        for i in range(self.spatial_size):
            embedded_inputs.append(self.emb_layers[i](inputs[i]))

        embedded_inputs = tf.concat(embedded_inputs, axis=1)

        return embedded_inputs


def main():
    # Temporal input with value, time, and null dimensions
    data = np.random.randn(100, 24, 5).astype(np.float32)
    data[:, :, 3] = np.arange(data.shape[1])
    null_max_size = np.max(data[:, :, -2]) + 1
    data[:, :, 4] = np.arange(data.shape[1])
    time_max_sizes = [np.max(data[:, :, -2]) + 1]
    feature_mask = [0, 0, 0, 1, 2]

    # Spatial input
    sdata = [data, data, data]

    # PositionalEmbedding init & call
    embedder = TemporalEmbedding(
        d_model=512,
        kernel_size=3,
        feature_mask=feature_mask,
        time_max_sizes=time_max_sizes,
        null_max_size=null_max_size
    )
    x_emb = embedder(data)
    print(f'TemporalEmbedding:  {x_emb.shape} {data.shape}')

    # GlobalEmbedding init & call
    global_embedder = SpatialEmbedding(
        d_model=512,
        kernel_size=3,
        spatial_size=len(sdata),
        feature_mask=[0, 0, 0, 0, 0]
    )
    sx_emb = global_embedder(sdata)
    print(f'GlobalEmbedding:  {sx_emb.shape}')
    print('Hello World!')


if __name__ == '__main__':
    main()
