
import tensorflow as tf
from tensorflow import keras

class TemporalSoftmaxPool(keras.layers.Layer):
    """Creates a network layer corresponding to temporal softmax pooling.

    This is useful for multi-class logits (used in e.g., Charades). Modified from
    AssembleNet Charades evaluation from:

    Michael S. Ryoo, AJ Piergiovanni, Mingxing Tan, Anelia Angelova.
    AssembleNet: Searching for Multi-Stream Neural Connectivity in Video
    Architectures.
    (https://arxiv.org/pdf/1905.13209.pdf).
    """

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        assert inputs.shape.rank in (3, 4, 5)
        frames = tf.shape(inputs)[1]
        pre_logits = inputs / tf.sqrt(tf.cast(frames, inputs.dtype))
        activations = tf.nn.softmax(pre_logits, axis=1)
        outputs = inputs * activations
        return outputs