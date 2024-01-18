
from tensorflow import keras
import tensorflow as tf

class Squeeze3D(keras.layers.Layer):
    """Squeeze3D layer to remove singular dimensions."""

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        return tf.squeeze(inputs, axis=(1, 2, 3))