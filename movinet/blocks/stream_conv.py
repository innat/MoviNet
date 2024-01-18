
from .conv import ConvBlock
from typing import Union, Sequence, Optional, Any, Tuple

import tensorflow as tf
from tensorflow import keras
from movinet.utils import normalize_tuple

from movinet.config import States
from movinet.layers.stream_buffer import StreamBuffer


KERNEL_WEIGHT_DECAY = 1.5e-5

class StreamConvBlock(ConvBlock):
    """ConvBlock with StreamBuffer."""

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
        state_prefix: Optional[
            str
        ] = None,  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Initializes a stream conv block.

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
          state_prefix: a prefix string to identify states.
          **kwargs: keyword arguments to be passed to this layer.

        Returns:
          A output tensor of the StreamConvBlock operation.
        """
        kernel_size = normalize_tuple(kernel_size, 3, "kernel_size")
        buffer_size = kernel_size[0] - 1
        use_buffer = buffer_size > 0 and causal

        self._state_prefix = state_prefix

        super(StreamConvBlock, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            depthwise=depthwise,
            causal=causal,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=use_batch_norm,
            batch_norm_layer=batch_norm_layer,
            batch_norm_momentum=batch_norm_momentum,
            batch_norm_epsilon=batch_norm_epsilon,
            use_sync_bn=use_sync_bn,
            activation=activation,
            conv_type=conv_type,
            use_buffered_input=use_buffer,
            **kwargs
        )

        self._stream_buffer = None
        if use_buffer:
            self._stream_buffer = StreamBuffer(
                buffer_size=buffer_size, state_prefix=state_prefix
            )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {"state_prefix": self._state_prefix}
        base_config = super(StreamConvBlock, self).get_config()
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

        x = inputs

        # If we have no separate temporal conv, use the buffer before the 3D conv.
        if self._conv_temporal is None and self._stream_buffer is not None:
            x, states = self._stream_buffer(x, states=states)

        # bn_op and activation_op are folded into the '2plus1d' conv layer so that
        # we do not explicitly call them here.
        # TODO(lzyuan): clean the conv layers api once the models are re-trained.
        x = self._conv(x)
        if self._batch_norm is not None and self._conv_type != "2plus1d":
            x = self._batch_norm(x)
        if self._activation_layer is not None and self._conv_type != "2plus1d":
            x = self._activation_layer(x)

        if self._conv_temporal is not None:
            if self._stream_buffer is not None:
                # If we have a separate temporal conv, use the buffer before the
                # 1D conv instead (otherwise, we may waste computation on the 2D conv).
                x, states = self._stream_buffer(x, states=states)

            x = self._conv_temporal(x)
            if self._batch_norm_temporal is not None and self._conv_type != "2plus1d":
                x = self._batch_norm_temporal(x)
            if self._activation_layer is not None and self._conv_type != "2plus1d":
                x = self._activation_layer(x)

        return x, states