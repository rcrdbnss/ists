import random

import numpy as np
import tensorflow as tf
import keras

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


def get_decay_factor(time_deltas):
    return 1 / tf.math.log(tf.math.exp(1.) + time_deltas)


def get_timedeltas_from_mask(mask: np.ndarray) -> np.ndarray:  # mask: [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    mask = mask.astype(bool)
    mask_shape = np.shape(mask)
    B, T = mask_shape[0], mask_shape[1]
    time_steps = np.arange(T)  # t: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    time_deltas = []
    for i in range(B):
        m = mask[i]
        t = time_steps[m]  # t: [0, 1, 3, 5, 6, 9]
        deltas = np.diff(t)  # deltas: [1, 2, 2, 1, 3]
        deltas -= 1  # deltas: [0, 1, 1, 0, 2]
        deltas = np.insert(deltas, 0, 0)  # deltas: [0, 0, 1, 1, 0, 2]
        # _time_deltas = np.zeros_like(time_steps)  # _time_deltas: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        _time_deltas = np.full(T, float('inf'))  # _time_deltas: [inf, inf, inf, inf, inf, inf, inf, inf, inf, inf]
        np.put_along_axis(_time_deltas, np.where(m)[0], deltas, 0)  # _time_deltas: [0*, 0*, 0, 1*, 0, 1*, 0*, 0, 0, 2*] (*: mask=True)
        time_deltas.append(_time_deltas)
    time_deltas = np.stack(time_deltas, dtype=float)
    return time_deltas


@keras.saving.register_keras_serializable()  # ???, required to work with bidirectional=True
class TGRUCell(tf.keras.layers.GRUCell):

    def __init__(
            self,
            units,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            **kwargs
    ):
        super().__init__(
            units,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            **kwargs
        )

    def call(self, inputs, states, training=False):
        # To pass an additional input to the cell, I must concatenate it to the input tensor
        time_deltas = inputs[:, -1]
        inputs = inputs[:, :-1]
        decay_factor = get_decay_factor(time_deltas)

        # h_tm1 = (
        #     states[0] if tree.is_nested(states) else states
        # )
        h_tm1 = states[0]
        h_tm1 = h_tm1 * tf.expand_dims(decay_factor, -1)
        # states = [h_tm1] if tree.is_nested(states) else h_tm1
        states = [h_tm1]
        return super().call(inputs, states, training=training)

    def build(self, input_shape):
        s0, s1 = input_shape
        s1 -= 1
        input_shape = (s0, s1)
        super().build(input_shape)

    def get_config(self):
        config = {
            "units": self.units,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TGRU(tf.keras.layers.Layer):

    def __init__(
            self,
            units,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            go_backwards=False,
            bidirectional=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.go_backwards = go_backwards

        cell = TGRUCell(
            units,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
        )

        # To make it work with Bidirectional, I must wrap the RNN inside this class rather than making this class extend RNN
        self.rnn = tf.keras.layers.RNN(
            cell,
            go_backwards=go_backwards,
            **kwargs
        )
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn)

    def call(self, sequences, initial_state=None, mask=None, training=False, time_deltas=None):
        # To pass an additional input to the cell, I must concatenate it to the input tensor
        if time_deltas is None:
            time_deltas = tf.zeros_like(sequences[:, :, 0:1])
        else:
            time_deltas = tf.expand_dims(time_deltas, axis=-1)
        sequences = tf.concat([sequences, time_deltas], axis=-1)
        return self.rnn(sequences, initial_state=initial_state, mask=mask, training=training)

    def build(self, sequences_shape):
        # Tensorflow maps call() arguments to build() arguments by adding a _shape suffix to the call() argument name
        s0, s1, s2 = sequences_shape
        s2 += 1
        sequences_shape = (s0, s1, s2)
        self.rnn.build(sequences_shape)

    def get_config(self):
        config = {
            "units": self.units,
            "kernel_regularizer": self.kernel_regularizer,
            "recurrent_regularizer": self.recurrent_regularizer,
            "go_backwards": self.go_backwards,
            "bidirectional": self.bidirectional
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



if __name__ == '__main__':
    t_gru = TGRU(
        256,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
    )

    B, T, F = 32, 20, 8

    # Input shape (batch_size, time_steps, features)
    input_data = np.random.randn(B, T, F)

    # Generate a randon mask in which 30% of its 48 values are set to False
    mask = np.random.choice([0, 1], size=(B, T), p=[0.3, 0.7])  # m: [1, 1, 0, 1, 0, 1, 1, 0, 0, 1]
    mask = mask.astype(bool)

    time_deltas = get_timedeltas_from_mask(mask)
    output = t_gru(input_data, mask=mask, time_deltas=time_deltas)

    print(output.shape)
