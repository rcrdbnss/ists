import numpy as np
import tensorflow as tf


class MeanPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MeanPooling, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            inputs *= mask_float
            sum_inputs = tf.reduce_sum(inputs, axis=-2)
            count = tf.reduce_sum(mask_float, axis=-2)
            avg = sum_inputs / (count + 1e-9)
        else:
            avg = tf.reduce_mean(inputs, axis=-2)
        return avg


class LastTokenPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LastTokenPooling, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is not None:
            mask_sum = tf.reduce_sum(mask, axis=-1)  # (batch,)
            last_token_indices = tf.maximum(mask_sum - 1, 0)  # (batch,)
            batch_indices = tf.range(tf.shape(inputs)[0])  # (batch,)
            gather_indices = tf.stack([batch_indices, last_token_indices], axis=1)  # (batch, 2)
            last_tokens = tf.gather_nd(inputs, gather_indices)  # (batch, d_model)
        else:
            last_tokens = inputs[:, -1]  # (batch, d_model)
        return last_tokens


class AttentivePooling(tf.keras.layers.Layer):
    """Attentive Pooling implementation using a Dense layer."""

    def __init__(self, **kwargs):
        super(AttentivePooling, self).__init__(**kwargs)
        self.scorer = None
        self.scale = None

    def build(self, input_shape):
        # input_shape is (batch, seq_len, d_model)
        d_model = input_shape[-1]

        # Define the scaling factor: 1 / sqrt(d_model)
        # self.scale = tf.math.rsqrt(tf.cast(d_model, tf.float32))
        self.scale = 1 / np.sqrt(d_model)

        # The query vector (kernel)
        self.scorer = self.add_weight(shape=(d_model, 1), name="attention_scorer")
        super(AttentivePooling, self).build(input_shape)

    def call(self, inputs, mask=None):
        # 1. Calculate raw dot products
        # Shape: (batch, seq_len, 1)
        scores = tf.matmul(inputs, self.scorer)

        # 2. APPLY SCALING (Critical for wider models)
        scores = scores * self.scale

        # 3. Apply mask
        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            scores += (1.0 - mask_float) * -1e9

        # 4. Softmax
        attention_weights = tf.nn.softmax(scores, axis=-2)

        # 5. Weighted Average
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=-2)

        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class AttentivePoolingWithPositionalBias(tf.keras.layers.Layer):
    """Attentive Pooling implementation using a Dense layer."""

    def __init__(self, **kwargs):
        super(AttentivePoolingWithPositionalBias, self).__init__(**kwargs)
        self.scorer = None
        self.scale = None
        self.pos_bias = None

    def build(self, input_shape):
        # input_shape is (batch, seq_len, d_model)
        _, seq_len, d_model = input_shape

        # Define the scaling factor: 1 / sqrt(d_model)
        self.scale = 1 / np.sqrt(d_model)

        # The query vector (kernel)
        self.scorer = tf.keras.layers.Dense(units=1, activation=None, name="attention_scorer", use_bias=False)

        # Positional embedding layer
        self.pos_bias = self.add_weight(name="pos_bias", shape=(seq_len, 1), initializer='zeros')

        super(AttentivePoolingWithPositionalBias, self).build(input_shape)

    def call(self, inputs, mask=None):
        k, v = inputs, inputs

        scores = self.scorer(k)

        scores = scores * self.scale
        scores = scores + self.pos_bias

        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            scores += (1.0 - mask_float) * -1e9

        attention_weights = tf.nn.softmax(scores, axis=-2)

        weighted_sum = tf.reduce_sum(attention_weights * v, axis=-2)

        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
