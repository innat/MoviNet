
import tenosrflow as tf 
from tensorflow import keras
from typing import Union, Optional, Tuple, Sequence
from movinet.config import States, Activation

from movinet.blocks.conv import ConvBlock
from movinet.blocks.stream_conv import StreamConvBlock
from movinet.layers.stream_squeeze_excitation import StreamSqueezeExcitation
from movinet.blocks.skip_block import SkipBlock
from movinet.blocks.mobile_bottleneck import MobileBottleneck

from movinet.utils import normalize_tuple, make_divisible

KERNEL_WEIGHT_DECAY = 1.5e-5

class MovinetBlock(keras.layers.Layer):
    """A basic block for MoViNets.

    Applies a mobile inverted bottleneck with pointwise expansion, 3D depthwise
    convolution, 3D squeeze excite, pointwise projection, and residual connection.
    """

    def __init__(
        self,
        out_filters: int,
        expand_filters: int,
        kernel_size: Union[int, Sequence[int]] = (3, 3, 3),
        strides: Union[int, Sequence[int]] = (1, 1, 1),
        causal: bool = False,
        activation: Activation = "swish",
        gating_activation: Activation = "sigmoid",
        se_ratio: float = 0.25,
        stochastic_depth_drop_rate: float = 0.0,
        conv_type: str = "3d",
        se_type: str = "3d",
        use_positional_encoding: bool = False,
        kernel_initializer: tf.keras.initializers.Initializer = "HeNormal",
        kernel_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = tf.keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
        batch_norm_layer: tf.keras.layers.Layer = tf.keras.layers.BatchNormalization,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        use_sync_bn: bool = False,
        state_prefix: Optional[
            str
        ] = None,  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Implementation for MoViNet block.

        Args:
          out_filters: number of output filters for the final projection.
          expand_filters: number of expansion filters after the input.
          kernel_size: kernel size of the main depthwise convolution.
          strides: strides of the main depthwise convolution.
          causal: if True, run the temporal convolutions in causal mode.
          activation: activation to use across all conv operations.
          gating_activation: gating activation to use in squeeze excitation layers.
          se_ratio: squeeze excite filters ratio.
          stochastic_depth_drop_rate: optional drop rate for stochastic depth.
          conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
              ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
              own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
              uses two sequential 3D ops instead.
          se_type: '3d', '2d', or '2plus3d'. '3d' uses the default 3D
              spatiotemporal global average pooling for squeeze excitation. '2d'
              uses 2D spatial global average pooling  on each frame. '2plus3d'
              concatenates both 3D and 2D global average pooling.
          use_positional_encoding: add a positional encoding after the (cumulative)
              global average pooling layer in the squeeze excite layer.
          kernel_initializer: kernel initializer for the conv operations.
          kernel_regularizer: kernel regularizer for the conv operations.
          batch_norm_layer: class to use for batch norm.
          batch_norm_momentum: momentum of the batch norm operation.
          batch_norm_epsilon: epsilon of the batch norm operation.
          use_sync_bn: if True, use synchronized batch normalization.
          state_prefix: a prefix string to identify states.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(MovinetBlock, self).__init__(**kwargs)

        self._kernel_size = normalize_tuple(kernel_size, 3, "kernel_size")
        self._strides = normalize_tuple(strides, 3, "strides")

        # Use a multiplier of 2 if concatenating multiple features
        se_multiplier = 2 if se_type == "2plus3d" else 1
        se_hidden_filters = make_divisible(
            se_ratio * expand_filters * se_multiplier, divisor=8
        )
        self._out_filters = out_filters
        self._expand_filters = expand_filters
        self._causal = causal
        self._activation = activation
        self._gating_activation = gating_activation
        self._se_ratio = se_ratio
        self._downsample = any(s > 1 for s in self._strides)
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._conv_type = conv_type
        self._se_type = se_type
        self._use_positional_encoding = use_positional_encoding
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._batch_norm_layer = batch_norm_layer
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._use_sync_bn = use_sync_bn
        self._state_prefix = state_prefix

        self._expansion = ConvBlock(
            expand_filters,
            (1, 1, 1),
            activation=activation,
            conv_type=conv_type,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=True,
            batch_norm_layer=self._batch_norm_layer,
            batch_norm_momentum=self._batch_norm_momentum,
            batch_norm_epsilon=self._batch_norm_epsilon,
            use_sync_bn=self._use_sync_bn,
            name="expansion",
        )
        self._feature = StreamConvBlock(
            expand_filters,
            self._kernel_size,
            strides=self._strides,
            depthwise=True,
            causal=self._causal,
            activation=activation,
            conv_type=conv_type,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=True,
            batch_norm_layer=self._batch_norm_layer,
            batch_norm_momentum=self._batch_norm_momentum,
            batch_norm_epsilon=self._batch_norm_epsilon,
            use_sync_bn=self._use_sync_bn,
            state_prefix=state_prefix,
            name="feature",
        )
        self._projection = ConvBlock(
            out_filters,
            (1, 1, 1),
            activation=None,
            conv_type=conv_type,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=True,
            batch_norm_layer=self._batch_norm_layer,
            batch_norm_momentum=self._batch_norm_momentum,
            batch_norm_epsilon=self._batch_norm_epsilon,
            use_sync_bn=self._use_sync_bn,
            name="projection",
        )
        self._attention = None
        if se_type != "none":
            self._attention = StreamSqueezeExcitation(
                se_hidden_filters,
                se_type=se_type,
                activation=activation,
                gating_activation=gating_activation,
                causal=self._causal,
                conv_type=conv_type,
                use_positional_encoding=use_positional_encoding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                state_prefix=state_prefix,
                name="se",
            )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "out_filters": self._out_filters,
            "expand_filters": self._expand_filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "causal": self._causal,
            "activation": self._activation,
            "gating_activation": self._gating_activation,
            "se_ratio": self._se_ratio,
            "stochastic_depth_drop_rate": self._stochastic_depth_drop_rate,
            "conv_type": self._conv_type,
            "se_type": self._se_type,
            "use_positional_encoding": self._use_positional_encoding,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "batch_norm_momentum": self._batch_norm_momentum,
            "batch_norm_epsilon": self._batch_norm_epsilon,
            "use_sync_bn": self._use_sync_bn,
            "state_prefix": self._state_prefix,
        }
        base_config = super(MovinetBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Builds the layer with the given input shape."""
        if input_shape[-1] == self._out_filters and not self._downsample:
            self._skip = None
        else:
            self._skip = SkipBlock(
                self._out_filters,
                downsample=self._downsample,
                conv_type=self._conv_type,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                name="skip",
            )

        self._mobile_bottleneck = MobileBottleneck(
            self._expansion,
            self._feature,
            self._projection,
            attention_layer=self._attention,
            skip_layer=self._skip,
            stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
            name="bneck",
        )

        super(MovinetBlock, self).build(input_shape)

    def call(
        self, inputs: tf.Tensor, states: Optional[States] = None
    ) -> Tuple[tf.Tensor, States]:
        """Calls the layer with the given inputs.

        Args:
          inputs: the input tensor.
          states: a dict of states such that, if any of the keys match for this
              layer, will overwrite the contents of the buffer(s).

        Returns:
          the output tensor and states
        """
        states = dict(states) if states is not None else {}
        return self._mobile_bottleneck(inputs, states=states)