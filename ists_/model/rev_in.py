import tensorflow as tf

class RevIN(tf.keras.layers.Layer):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = self.add_weight(shape=(1, 1, num_features),
                                         initializer="ones",
                                         trainable=True,
                                         name="gamma")
            self.beta = self.add_weight(shape=(1, 1, num_features),
                                        initializer="zeros",
                                        trainable=True,
                                        name="beta")

    def compute_masked_stats(self, x, mask):
        """
        Compute masked mean and std over time dimension.
        """
        mask = tf.cast(mask, x.dtype)  # (batch, time)
        # mask = tf.expand_dims(mask, axis=-1)  # (batch, time, 1)

        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        count = tf.maximum(count, 1.0)  # avoid divide-by-zero

        mean = tf.reduce_sum(x * mask, axis=1, keepdims=True) / count
        var = tf.reduce_sum(((x - mean) * mask) ** 2, axis=1, keepdims=True) / count
        std = tf.sqrt(var + self.eps)

        return mean, std

    def normalize(self, x, mask):
        """
        Normalize with respect to instance (window) statistics.
        Returns:
            x_norm: normalized tensor (batch, time, features)
            (mean, std): stats for denormalization
        """
        mean, std = self.compute_masked_stats(x, mask)
        x_norm = (x - mean) / std

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm, (mean, std)

    def denormalize(self, x, stats):
        """
        Denormalize using given stats (mean, std).
        Args:
            x: tensor (batch, ...)
            stats: (mean, std), each of shape (batch, 1, features)

        Returns:
            denormalized tensor
        """
        mean, std = stats

        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)

        return x * std + mean

    def call(self, x, *, mode, **kwargs):
        """
        Args:
            x: input tensor (batch, time, features)
            mode: 'norm' or 'denorm'
            **kwargs: additional arguments (e.g., mask)

        Returns:
            normalized or denormalized tensor
        """

        if mode == 'norm':
            mask = kwargs.get('mask', tf.ones_like(x, dtype=tf.float32))
            return self.normalize(x, mask)
        elif mode == 'denorm':
            stats = kwargs.get('stats')
            if stats is None:
                raise ValueError("Stats must be provided for denormalization.")
            return self.denormalize(x, stats)
        else:
            raise ValueError("Mode must be either 'norm' or 'denorm'.")
