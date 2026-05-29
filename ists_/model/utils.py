import tensorflow as tf


class SwiGLUDense(tf.keras.layers.Layer):
    def __init__(self, units, inputs, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.V = tf.keras.layers.Dense(
            units, activation=None, kernel_regularizer=self.kernel_regularizer
        )
        self.G = tf.keras.layers.Dense(
            units, activation=None, kernel_regularizer=self.kernel_regularizer,
            bias_initializer='ones'
        )

    def call(self, x):
        v = self.V(x)
        g = self.G(x)
        x = v * tf.nn.silu(g)
        return x


class SwiGLUConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', l2_reg=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None

        self.V = tf.keras.layers.Conv1D(
            filters, kernel_size, padding=padding, activation=None, kernel_regularizer=self.l2_reg
        )
        self.G = tf.keras.layers.Conv1D(
            filters, kernel_size, padding=padding, activation=None, kernel_regularizer=self.l2_reg,
            bias_initializer='ones'
        )

    def call(self, x):
        v = self.V(x)
        g = self.G(x)
        x = v * tf.nn.silu(g)
        return x
