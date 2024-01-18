from .utils import CausalConvMixin
import tensorflow as tf
from tensorflow import keras

from typing import List

class DepthwiseConv2D(keras.layers.DepthwiseConv2D, CausalConvMixin):
    """DepthwiseConv2D layer supporting CausalConv.

    Supports `padding='causal'` option (like in `tf_keras.layers.Conv1D`),
    which applies causal padding to the temporal dimension, and same padding in
    the spatial dimensions.
    """

    def __init__(self, *args, use_buffered_input=False, **kwargs):
        """Initializes depthwise conv2d.

        Args:
          *args: Arguments to be passed.
          use_buffered_input: A `bool`. If True, the input is expected to be padded
            beforehand. In effect, calling this layer will use 'valid' padding on
            the temporal dimension to simulate 'causal' padding.
          **kwargs: Additional keyword arguments to be passed.

        Returns:
          An output `tf.Tensor` of the DepthwiseConv2D operation.
        """
        super(DepthwiseConv2D, self).__init__(*args, **kwargs)
        self._use_buffered_input = use_buffered_input

        # Causal padding is unsupported by default for DepthwiseConv2D,
        # so we resort to valid padding internally. However, we handle
        # causal padding as a special case with `self._is_causal`, which is
        # defined by the super class.
        if self.padding == "causal":
            self.padding = "valid"

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "use_buffered_input": self._use_buffered_input,
        }
        base_config = super(DepthwiseConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        if self._is_causal:
            inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
        return super(DepthwiseConv2D, self).call(inputs)

    def _compute_causal_padding(self, inputs):
        """Computes causal padding dimensions for the given inputs."""
        return self._compute_buffered_causal_padding(
            inputs, use_buffered_input=self._use_buffered_input
        )

    def _validate_init(self):
        """Validates the Conv layer initial configuration."""
        self._causal_validate_init()

    def _spatial_output_shape(self, spatial_input_shape: List[int]):
        """Computes the spatial output shape from the input shape."""
        shape = super(DepthwiseConv2D, self)._spatial_output_shape(spatial_input_shape)
        return self._buffered_spatial_output_shape(shape)