

import tensorflow as tf 
from tensorflow import keras
from movinet.config import States, Activation
from typing import Optional, Tuple
from movinet.blocks.conv import ConvBlock
from movinet.layers.gap3 import GlobalAveragePool3D
from movinet.layers.sap3 import SpatialAveragePool3D
from movinet.layers.positional_embedding import PositionalEncoding

KERNEL_WEIGHT_DECAY = 1.5e-5

class StreamSqueezeExcitation(keras.layers.Layer):
    """Squeeze and excitation layer with causal mode.

    Reference: https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(
        self,
        hidden_filters: int,
        se_type: str = "3d",
        activation: Activation = "swish",
        gating_activation: Activation = "sigmoid",
        causal: bool = False,
        conv_type: str = "3d",
        kernel_initializer: keras.initializers.Initializer = "HeNormal",
        kernel_regularizer: Optional[
            keras.regularizers.Regularizer
        ] = keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
        use_positional_encoding: bool = False,
        state_prefix: Optional[
            str
        ] = None,  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Implementation for squeeze and excitation.

        Args:
          hidden_filters: The hidden filters of squeeze excite.
          se_type: '3d', '2d', or '2plus3d'. '3d' uses the default 3D
              spatiotemporal global average pooling for squeeze excitation. '2d'
              uses 2D spatial global average pooling  on each frame. '2plus3d'
              concatenates both 3D and 2D global average pooling.
          activation: name of the activation function.
          gating_activation: name of the activation function for gating.
          causal: if True, use causal mode in the global average pool.
          conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
              ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
              own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
              uses two sequential 3D ops instead.
          kernel_initializer: kernel initializer for the conv operations.
          kernel_regularizer: kernel regularizer for the conv operation.
          use_positional_encoding: add a positional encoding after the (cumulative)
              global average pooling layer.
          state_prefix: a prefix string to identify states.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(StreamSqueezeExcitation, self).__init__(**kwargs)

        self._hidden_filters = hidden_filters
        self._se_type = se_type
        self._activation = activation
        self._gating_activation = gating_activation
        self._causal = causal
        self._conv_type = conv_type
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._use_positional_encoding = use_positional_encoding
        self._state_prefix = state_prefix

        self._spatiotemporal_pool = GlobalAveragePool3D(
            keepdims=True, causal=causal, state_prefix=state_prefix
        )
        self._spatial_pool = SpatialAveragePool3D(keepdims=True)

        self._pos_encoding = None
        if use_positional_encoding:
            self._pos_encoding = PositionalEncoding(
                initializer="zeros", state_prefix=state_prefix
            )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "hidden_filters": self._hidden_filters,
            "se_type": self._se_type,
            "activation": self._activation,
            "gating_activation": self._gating_activation,
            "causal": self._causal,
            "conv_type": self._conv_type,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "use_positional_encoding": self._use_positional_encoding,
            "state_prefix": self._state_prefix,
        }
        base_config = super(StreamSqueezeExcitation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        """Builds the layer with the given input shape."""
        self._se_reduce = ConvBlock(
            filters=self._hidden_filters,
            kernel_size=1,
            causal=self._causal,
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_batch_norm=False,
            activation=self._activation,
            conv_type=self._conv_type,
            name="se_reduce",
        )

        self._se_expand = ConvBlock(
            filters=input_shape[-1],
            kernel_size=1,
            causal=self._causal,
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_batch_norm=False,
            activation=self._gating_activation,
            conv_type=self._conv_type,
            name="se_expand",
        )

        super(StreamSqueezeExcitation, self).build(input_shape)

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

        if self._se_type == "3d":
            x, states = self._spatiotemporal_pool(
                inputs, states=states, output_states=True
            )
        elif self._se_type == "2d":
            x = self._spatial_pool(inputs)
        elif self._se_type == "2plus3d":
            x_space = self._spatial_pool(inputs)
            x, states = self._spatiotemporal_pool(
                x_space, states=states, output_states=True
            )

            if not self._causal:
                x = tf.tile(x, [1, tf.shape(inputs)[1], 1, 1, 1])

            x = tf.concat([x, x_space], axis=-1)
        else:
            raise ValueError("Unknown Squeeze Excitation type {}".format(self._se_type))

        if self._pos_encoding is not None:
            x, states = self._pos_encoding(x, states=states)

        x = self._se_reduce(x)
        x = self._se_expand(x)

        return x * inputs, states