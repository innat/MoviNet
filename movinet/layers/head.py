

import tensorflow as tf
from tensorflow import keras
from typing import Union, Optional, Tuple, Mapping
from movinet.config import States, Activation
from movinet.blocks.conv import ConvBlock
from movinet.layers.gap3 import GlobalAveragePool3D
from movinet.layers.sap3 import SpatialAveragePool3D

KERNEL_WEIGHT_DECAY = 1.5e-5

class Head(keras.layers.Layer):
    """Head layer for video networks.

    Applies pointwise projection and global pooling.
    """

    def __init__(
        self,
        project_filters: int,
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
        average_pooling_type: str = "3d",
        state_prefix: Optional[
            str
        ] = None,  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Implementation for video model head.

        Args:
          project_filters: number of pointwise projection filters.
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
          average_pooling_type: The average pooling type. Currently supporting
            ['3d', '2d', 'none'].
          state_prefix: a prefix string to identify states.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(Head, self).__init__(**kwargs)

        self._project_filters = project_filters
        self._conv_type = conv_type
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._batch_norm_layer = batch_norm_layer
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._use_sync_bn = use_sync_bn
        self._state_prefix = state_prefix

        self._project = ConvBlock(
            filters=project_filters,
            kernel_size=1,
            activation=activation,
            conv_type=conv_type,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=True,
            batch_norm_layer=self._batch_norm_layer,
            batch_norm_momentum=self._batch_norm_momentum,
            batch_norm_epsilon=self._batch_norm_epsilon,
            use_sync_bn=self._use_sync_bn,
            name="project",
        )
        if average_pooling_type.lower() == "3d":
            self._pool = GlobalAveragePool3D(
                keepdims=True, causal=False, state_prefix=state_prefix
            )
        elif average_pooling_type.lower() == "2d":
            self._pool = SpatialAveragePool3D(keepdims=True)
        elif average_pooling_type == "none":
            self._pool = None
        else:
            raise ValueError(
                "%s average_pooling_type is not supported." % average_pooling_type
            )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "project_filters": self._project_filters,
            "conv_type": self._conv_type,
            "activation": self._activation,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "batch_norm_momentum": self._batch_norm_momentum,
            "batch_norm_epsilon": self._batch_norm_epsilon,
            "use_sync_bn": self._use_sync_bn,
            "state_prefix": self._state_prefix,
        }
        base_config = super(Head, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(
        self,
        inputs: Union[tf.Tensor, Mapping[str, tf.Tensor]],
        states: Optional[States] = None,
    ) -> Tuple[tf.Tensor, States]:
        """Calls the layer with the given inputs.

        Args:
          inputs: the input tensor or dict of endpoints.
          states: a dict of states such that, if any of the keys match for this
              layer, will overwrite the contents of the buffer(s).

        Returns:
          the output tensor and states
        """
        states = dict(states) if states is not None else {}
        x = self._project(inputs)
        if self._pool is not None:
            outputs = self._pool(x, states=states, output_states=True)
        else:
            outputs = (x, states)
        return outputs