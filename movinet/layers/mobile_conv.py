import tensorflow as tf
from tensorflow import keras

from typing import Union, Sequence, Optional, Any
from .conv import Conv2D
from .depthwise_conv import DepthwiseConv2D
from movinet.utils import normalize_tuple



class MobileConv2D(keras.layers.Layer):
    """Conv2D layer with extra options to support mobile devices.

    Reshapes 5D video tensor inputs to 4D, allowing Conv2D to run across
    dimensions (2, 3) or (3, 4). Reshapes tensors back to 5D when returning the
    output.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = (1, 1),
        padding: str = "valid",
        data_format: Optional[str] = None,
        dilation_rate: Union[int, Sequence[int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[keras.constraints.Constraint] = None,
        bias_constraint: Optional[keras.constraints.Constraint] = None,
        use_depthwise: bool = False,
        use_temporal: bool = False,
        use_buffered_input: bool = False,  # pytype: disable=annotation-type-mismatch  # typed-keras
        batch_norm_op: Optional[Any] = None,
        activation_op: Optional[Any] = None,
        **kwargs
    ):  # pylint: disable=g-doc-args
        """Initializes mobile conv2d.

        For the majority of arguments, see tf_keras.layers.Conv2D.

        Args:
          use_depthwise: if True, use DepthwiseConv2D instead of Conv2D
          use_temporal: if True, apply Conv2D starting from the temporal dimension
              instead of the spatial dimensions.
          use_buffered_input: if True, the input is expected to be padded
              beforehand. In effect, calling this layer will use 'valid' padding on
              the temporal dimension to simulate 'causal' padding.
          batch_norm_op: A callable object of batch norm layer. If None, no batch
            norm will be applied after the convolution.
          activation_op: A callabel object of activation layer. If None, no
            activation will be applied after the convolution.
          **kwargs: keyword arguments to be passed to this layer.

        Returns:
          A output tensor of the MobileConv2D operation.
        """
        super(MobileConv2D, self).__init__(**kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._data_format = data_format
        self._dilation_rate = dilation_rate
        self._groups = groups
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._activity_regularizer = activity_regularizer
        self._kernel_constraint = kernel_constraint
        self._bias_constraint = bias_constraint
        self._use_depthwise = use_depthwise
        self._use_temporal = use_temporal
        self._use_buffered_input = use_buffered_input
        self._batch_norm_op = batch_norm_op
        self._activation_op = activation_op

        kernel_size = normalize_tuple(kernel_size, 2, "kernel_size")

        if self._use_temporal and kernel_size[1] > 1:
            raise ValueError("Temporal conv with spatial kernel is not supported.")

        if use_depthwise:
            self._conv = DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                depth_multiplier=1,
                data_format=data_format,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                depthwise_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                depthwise_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                depthwise_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                use_buffered_input=use_buffered_input,
            )
        else:
            self._conv = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                groups=groups,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                use_buffered_input=use_buffered_input,
            )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "padding": self._padding,
            "data_format": self._data_format,
            "dilation_rate": self._dilation_rate,
            "groups": self._groups,
            "use_bias": self._use_bias,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "activity_regularizer": self._activity_regularizer,
            "kernel_constraint": self._kernel_constraint,
            "bias_constraint": self._bias_constraint,
            "use_depthwise": self._use_depthwise,
            "use_temporal": self._use_temporal,
            "use_buffered_input": self._use_buffered_input,
        }
        base_config = super(MobileConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        if self._use_temporal:
            input_shape = [
                tf.shape(inputs)[0],
                tf.shape(inputs)[1],
                tf.shape(inputs)[2] * tf.shape(inputs)[3],
                inputs.shape[4],
            ]
        else:
            input_shape = [
                tf.shape(inputs)[0] * tf.shape(inputs)[1],
                tf.shape(inputs)[2],
                tf.shape(inputs)[3],
                inputs.shape[4],
            ]
        x = tf.reshape(inputs, input_shape)

        x = self._conv(x)
        if self._batch_norm_op is not None:
            x = self._batch_norm_op(x)
        if self._activation_op is not None:
            x = self._activation_op(x)

        if self._use_temporal:
            output_shape = [
                tf.shape(x)[0],
                tf.shape(x)[1],
                tf.shape(inputs)[2],
                tf.shape(inputs)[3],
                x.shape[3],
            ]
        else:
            output_shape = [
                tf.shape(inputs)[0],
                tf.shape(inputs)[1],
                tf.shape(x)[1],
                tf.shape(x)[2],
                x.shape[3],
            ]
        x = tf.reshape(x, output_shape)

        return x