
import tensorflow as tf
from tensorflow import keras

class SpatialAveragePool3D(keras.layers.Layer):
    """Creates a global average pooling layer pooling across spatial dimentions."""

    def __init__(self, keepdims: bool = False, **kwargs):
        """Initializes a global average pool layer.

        Args:
          keepdims: A `bool`. If True, keep the averaged dimensions.
          **kwargs: Additional keyword arguments to be passed to this layer.

        Returns:
          An output `tf.Tensor`.
        """
        super(SpatialAveragePool3D, self).__init__(**kwargs)
        self._keepdims = keepdims

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "keepdims": self._keepdims,
        }
        base_config = super(SpatialAveragePool3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Builds the layer with the given input shape."""
        if tf.keras.backend.image_data_format() == "channels_first":
            raise ValueError('"channels_first" mode is unsupported.')

        super(SpatialAveragePool3D, self).build(input_shape)

    def call(self, inputs, states=None, output_states: bool = False):
        """Calls the layer with the given inputs."""
        if inputs.shape.rank != 5:
            raise ValueError(
                "Input should have rank {}, got {}".format(5, inputs.shape.rank)
            )

        output = tf.reduce_mean(inputs, axis=(2, 3), keepdims=self._keepdims)
        return (output, states) if output_states else output