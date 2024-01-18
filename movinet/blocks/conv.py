
from tensorflow import keras
from typing import Union, Sequence, Optional, Any

from movinet.utils import normalize_tuple, get_activation


from movinet.layers.conv import Conv3D
from movinet.layers.mobile_conv import MobileConv2D

KERNEL_WEIGHT_DECAY = 1.5e-5

class ConvBlock(keras.layers.Layer):
    """A Conv followed by optional BatchNorm and Activation."""

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = 1,
        depthwise: bool = False,
        causal: bool = False,
        use_bias: bool = False,
        kernel_initializer: keras.initializers.Initializer = "HeNormal",
        kernel_regularizer: Optional[
            keras.regularizers.Regularizer
        ] = keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
        use_batch_norm: bool = True,
        batch_norm_layer: keras.layers.Layer = keras.layers.BatchNormalization,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        use_sync_bn: bool = False,
        activation: Optional[Any] = None,
        conv_type: str = "3d",
        use_buffered_input: bool = False,  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Initializes a conv block.

        Args:
          filters: filters for the conv operation.
          kernel_size: kernel size for the conv operation.
          strides: strides for the conv operation.
          depthwise: if True, use DepthwiseConv2D instead of Conv2D
          causal: if True, use causal mode for the conv operation.
          use_bias: use bias for the conv operation.
          kernel_initializer: kernel initializer for the conv operation.
          kernel_regularizer: kernel regularizer for the conv operation.
          use_batch_norm: if True, apply batch norm after the conv operation.
          batch_norm_layer: class to use for batch norm, if applied.
          batch_norm_momentum: momentum of the batch norm operation, if applied.
          batch_norm_epsilon: epsilon of the batch norm operation, if applied.
          use_sync_bn: if True, use synchronized batch normalization.
          activation: activation after the conv and batch norm operations.
          conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
              ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
              own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
              uses two sequential 3D ops instead.
          use_buffered_input: if True, the input is expected to be padded
              beforehand. In effect, calling this layer will use 'valid' padding on
              the temporal dimension to simulate 'causal' padding.
          **kwargs: keyword arguments to be passed to this layer.

        Returns:
          A output tensor of the ConvBlock operation.
        """

        super(ConvBlock, self).__init__(**kwargs)

        kernel_size = normalize_tuple(kernel_size, 3, "kernel_size")
        strides = normalize_tuple(strides, 3, "strides")

        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._depthwise = depthwise
        self._causal = causal
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._use_batch_norm = use_batch_norm
        self._batch_norm_layer = batch_norm_layer
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._conv_type = conv_type
        self._use_buffered_input = use_buffered_input

        if activation is not None:
            self._activation_layer = get_activation(activation, use_keras_layer=True)
        else:
            self._activation_layer = None

        self._groups = None

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "depthwise": self._depthwise,
            "causal": self._causal,
            "use_bias": self._use_bias,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "use_batch_norm": self._use_batch_norm,
            "batch_norm_momentum": self._batch_norm_momentum,
            "batch_norm_epsilon": self._batch_norm_epsilon,
            "use_sync_bn": self._use_sync_bn,
            "activation": self._activation,
            "conv_type": self._conv_type,
            "use_buffered_input": self._use_buffered_input,
        }
        base_config = super(ConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Builds the layer with the given input shape."""
        padding = "causal" if self._causal else "same"
        self._groups = input_shape[-1] if self._depthwise else 1

        self._batch_norm = None
        self._batch_norm_temporal = None
        if self._use_batch_norm:
            self._batch_norm = self._batch_norm_layer(
                momentum=self._batch_norm_momentum,
                epsilon=self._batch_norm_epsilon,
                synchronized=self._use_sync_bn,
                name="bn",
            )
            if self._conv_type != "3d" and self._kernel_size[0] > 1:
                self._batch_norm_temporal = self._batch_norm_layer(
                    momentum=self._batch_norm_momentum,
                    epsilon=self._batch_norm_epsilon,
                    synchronized=self._use_sync_bn,
                    name="bn_temporal",
                )

        self._conv_temporal = None
        if self._conv_type == "3d_2plus1d" and self._kernel_size[0] > 1:
            self._conv = Conv3D(
                self._filters,
                (1, self._kernel_size[1], self._kernel_size[2]),
                strides=(1, self._strides[1], self._strides[2]),
                padding="same",
                groups=self._groups,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                use_buffered_input=False,
                name="conv3d",
            )
            self._conv_temporal = Conv3D(
                self._filters,
                (self._kernel_size[0], 1, 1),
                strides=(self._strides[0], 1, 1),
                padding=padding,
                groups=self._groups,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                use_buffered_input=self._use_buffered_input,
                name="conv3d_temporal",
            )
        elif self._conv_type == "2plus1d":
            self._conv = MobileConv2D(
                self._filters,
                (self._kernel_size[1], self._kernel_size[2]),
                strides=(self._strides[1], self._strides[2]),
                padding="same",
                use_depthwise=self._depthwise,
                groups=self._groups,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                use_buffered_input=False,
                batch_norm_op=self._batch_norm,
                activation_op=self._activation_layer,
                name="conv2d",
            )
            if self._kernel_size[0] > 1:
                self._conv_temporal = MobileConv2D(
                    self._filters,
                    (self._kernel_size[0], 1),
                    strides=(self._strides[0], 1),
                    padding=padding,
                    use_temporal=True,
                    use_depthwise=self._depthwise,
                    groups=self._groups,
                    use_bias=self._use_bias,
                    kernel_initializer=self._kernel_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    use_buffered_input=self._use_buffered_input,
                    batch_norm_op=self._batch_norm_temporal,
                    activation_op=self._activation_layer,
                    name="conv2d_temporal",
                )
        else:
            self._conv = Conv3D(
                self._filters,
                self._kernel_size,
                strides=self._strides,
                padding=padding,
                groups=self._groups,
                use_bias=self._use_bias,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                use_buffered_input=self._use_buffered_input,
                name="conv3d",
            )

        super(ConvBlock, self).build(input_shape)

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        x = inputs

        # bn_op and activation_op are folded into the '2plus1d' conv layer so that
        # we do not explicitly call them here.
        # TODO(lzyuan): clean the conv layers api once the models are re-trained.
        x = self._conv(x)
        if self._batch_norm is not None and self._conv_type != "2plus1d":
            x = self._batch_norm(x)
        if self._activation_layer is not None and self._conv_type != "2plus1d":
            x = self._activation_layer(x)

        if self._conv_temporal is not None:
            x = self._conv_temporal(x)
            if self._batch_norm_temporal is not None and self._conv_type != "2plus1d":
                x = self._batch_norm_temporal(x)
            if self._activation_layer is not None and self._conv_type != "2plus1d":
                x = self._activation_layer(x)

        return x