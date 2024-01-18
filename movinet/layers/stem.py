
import tensorflow as tf
from tensorflow import keras
from typing import Union, Optional, Tuple, Sequence
from movinet.config import States, Activation
from movinet.utils import normalize_tuple
from movinet.blocks.stream_conv import StreamConvBlock


KERNEL_WEIGHT_DECAY = 1.5e-5


class Stem(keras.layers.Layer):
    """Stem layer for video networks.

    Applies an initial convolution block operation.
    """

    def __init__(
        self,
        out_filters: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = (1, 1, 1),
        causal: bool = False,
        conv_type: str = "3d",
        activation: Activation = "swish",
        kernel_initializer: keras.initializers.Initializer = "HeNormal",
        kernel_regularizer: Optional[
            keras.regularizers.Regularizer
        ] = keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
        batch_norm_layer: keras.layers.Layer = keras.layers.BatchNormalization,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,
        use_sync_bn: bool = False,
        state_prefix: Optional[
            str
        ] = None,  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Implementation for video model stem.

        Args:
          out_filters: number of output filters.
          kernel_size: kernel size of the convolution.
          strides: strides of the convolution.
          causal: if True, run the temporal convolutions in causal mode.
          conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
              ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
              own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
              uses two sequential 3D ops instead.
          activation: the input activation name.
          kernel_initializer: kernel initializer for the conv operations.
          kernel_regularizer: kernel regularizer for the conv operations.
          batch_norm_layer: class to use for batch norm.
          batch_norm_momentum: momentum of the batch norm operation.
          batch_norm_epsilon: epsilon of the batch norm operation.
          use_sync_bn: if True, use synchronized batch normalization.
          state_prefix: a prefix string to identify states.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(Stem, self).__init__(**kwargs)

        self._out_filters = out_filters
        self._kernel_size = normalize_tuple(kernel_size, 3, "kernel_size")
        self._strides = normalize_tuple(strides, 3, "strides")
        self._causal = causal
        self._conv_type = conv_type
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._batch_norm_layer = batch_norm_layer
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._use_sync_bn = use_sync_bn
        self._state_prefix = state_prefix

        self._stem = StreamConvBlock(
            filters=self._out_filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            causal=self._causal,
            activation=self._activation,
            conv_type=self._conv_type,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_batch_norm=True,
            batch_norm_layer=self._batch_norm_layer,
            batch_norm_momentum=self._batch_norm_momentum,
            batch_norm_epsilon=self._batch_norm_epsilon,
            use_sync_bn=self._use_sync_bn,
            state_prefix=self._state_prefix,
            name="stem",
        )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "out_filters": self._out_filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "causal": self._causal,
            "activation": self._activation,
            "conv_type": self._conv_type,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "batch_norm_momentum": self._batch_norm_momentum,
            "batch_norm_epsilon": self._batch_norm_epsilon,
            "use_sync_bn": self._use_sync_bn,
            "state_prefix": self._state_prefix,
        }
        base_config = super(Stem, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        return self._stem(inputs, states=states)