import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


'''class AttentivePooling(Layer):
    """
    Implements an Attentive Pooling layer.

    This layer learns a context vector (query) and uses it to compute a weighted
    average of the input sequence, focusing on the most relevant time steps.

    Args:
        name (str): Name of the layer.
    """

    def __init__(self, **kwargs):
        super(AttentivePooling, self).__init__(**kwargs)
        self.query_vector = None

    def build(self, input_shape):
        """
        Creates the learnable weights of the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        feature_dim = input_shape[-1]

        # Create a learnable query vector. This vector will learn to identify
        # the most relevant parts of the sequence.
        self.query_vector = self.add_weight(
            name='pooling_query',
            shape=(feature_dim, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentivePooling, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        Forward pass of the layer.

        Args:
            inputs (tf.Tensor): The input sequence from the previous layer.
                                Shape: (batch_size, sequence_length, features).
            mask (tf.Tensor, optional): A boolean mask for handling padded sequences.
                                        Shape: (batch_size, sequence_length).

        Returns:
            tf.Tensor: The pooled output vector.
                       Shape: (batch_size, features).
        """
        # 1. Calculate the attention scores by taking the dot product of the
        #    input sequence with the learnable query vector.
        #    inputs shape: (batch, seq_len, features)
        #    query_vector shape: (features, 1)
        #    scores shape: (batch, seq_len, 1)
        scores = tf.tensordot(inputs, self.query_vector, axes=[[2], [0]])

        # 2. Apply the mask before softmax. This prevents the model from
        #    attending to padded time steps.
        if mask is not None:
            # The mask is boolean, so we invert it and multiply by a large
            # negative number to make the scores of padded steps negligible.
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            scores += (1.0 - mask_float) * -1e9

        # 3. Calculate the attention weights by applying softmax.
        #    The weights will sum to 1 for each sequence in the batch.
        #    weights shape: (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)

        # 4. Compute the weighted average of the input sequence.
        #    This creates the final pooled vector by multiplying the input
        #    by the learned importance weights.
        #    weighted_sum shape: (batch, features)
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=1)

        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])'''


class AttentivePooling(Layer):
    """Attentive Pooling implementation using a Dense layer."""

    def __init__(self, **kwargs):
        super(AttentivePooling, self).__init__(**kwargs)
        self.scorer = None

    def build(self, input_shape):
        # The Dense layer's kernel will act as our query vector.
        # We use no activation here because softmax is applied later.
        self.scorer = Dense(units=1, activation=None, name="attention_scorer")
        super(AttentivePooling, self).build(input_shape)

    def call(self, inputs, mask=None):
        # 1. Calculate scores using the Dense layer.
        #    The internal calculation is matmul(inputs, kernel) + bias
        #    Output shape: (batch, seq_len, 1)
        scores = self.scorer(inputs)

        # 2. Apply mask (same as before)
        if mask is not None:
            mask_float = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            scores += (1.0 - mask_float) * -1e9

        # 3. Calculate weights with softmax (same as before)
        attention_weights = tf.nn.softmax(scores, axis=1)

        # 4. Compute weighted average (same as before)
        weighted_sum = tf.reduce_sum(attention_weights * inputs, axis=1)

        return weighted_sum

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
