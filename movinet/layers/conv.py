from .utils import CausalConvMixin
import tensorflow as tf
from tensorflow import keras

from typing import List

class Conv2D(keras.layers.Conv2D, CausalConvMixin):
    """Conv2D layer supporting CausalConv.

    Supports `padding='causal'` option (like in `tf_keras.layers.Conv1D`),
    which applies causal padding to the temporal dimension, and same padding in
    the spatial dimensions.
    """

    def __init__(self, *args, use_buffered_input=False, **kwargs):
        """Initializes conv2d.

        Args:
          *args: Arguments to be passed.
          use_buffered_input: A `bool`. If True, the input is expected to be padded
            beforehand. In effect, calling this layer will use 'valid' padding on
            the temporal dimension to simulate 'causal' padding.
          **kwargs: Additional keyword arguments to be passed.

        Returns:
          An output `tf.Tensor` of the Conv2D operation.
        """
        super(Conv2D, self).__init__(*args, **kwargs)
        self._use_buffered_input = use_buffered_input

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "use_buffered_input": self._use_buffered_input,
        }
        base_config = super(Conv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        shape = super(Conv2D, self)._spatial_output_shape(spatial_input_shape)
        return self._buffered_spatial_output_shape(shape)
    

class Conv3D(keras.layers.Conv3D, CausalConvMixin):
    """Conv3D layer supporting CausalConv.

    Supports `padding='causal'` option (like in `tf_keras.layers.Conv1D`),
    which applies causal padding to the temporal dimension, and same padding in
    the spatial dimensions.
    """

    def __init__(self, *args, use_buffered_input=False, **kwargs):
        """Initializes conv3d.

        Args:
          *args: Arguments to be passed.
          use_buffered_input: A `bool`. If True, the input is expected to be padded
            beforehand. In effect, calling this layer will use 'valid' padding on
            the temporal dimension to simulate 'causal' padding.
          **kwargs: Additional keyword arguments to be passed.

        Returns:
          An output `tf.Tensor` of the Conv3D operation.
        """
        super(Conv3D, self).__init__(*args, **kwargs)
        self._use_buffered_input = use_buffered_input

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "use_buffered_input": self._use_buffered_input,
        }
        base_config = super(Conv3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Call the layer with the given inputs."""
        # Note: tf.nn.conv3d with depthwise kernels on CPU is currently only
        # supported when compiling with TF graph (XLA) using tf.function, so it
        # is compiled by default here (b/186463870).
        conv_fn = tf.function(super(Conv3D, self).call, jit_compile=True)
        return conv_fn(inputs)

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
        shape = super(Conv3D, self)._spatial_output_shape(spatial_input_shape)
        return self._buffered_spatial_output_shape(shape)
    

