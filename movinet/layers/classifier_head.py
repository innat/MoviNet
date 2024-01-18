import tensorflow as tf
from tensorflow import keras
from typing import Optional
from movinet.config import Activation
from movinet.blocks.conv import ConvBlock
from movinet.layers.temporal_softmax_pool import TemporalSoftmaxPool
from movinet.layers.squeeze3 import Squeeze3D

KERNEL_WEIGHT_DECAY = 1.5e-5

class ClassifierHead(keras.layers.Layer):
    """Head layer for video networks.

    Applies dense projection, dropout, and classifier projection. Expects input
    to be pooled vector with shape [batch_size, 1, 1, 1, num_channels]
    """

    def __init__(
        self,
        head_filters: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        conv_type: str = "3d",
        activation: Activation = "swish",
        output_activation: Optional[Activation] = None,
        max_pool_predictions: bool = False,
        kernel_initializer: keras.initializers.Initializer = "HeNormal",
        kernel_regularizer: Optional[
            keras.regularizers.Regularizer
        ] = keras.regularizers.L2(
            KERNEL_WEIGHT_DECAY
        ),  # pytype: disable=annotation-type-mismatch  # typed-keras
        **kwargs
    ):
        """Implementation for video model classifier head.

        Args:
          head_filters: number of dense head projection filters.
          num_classes: number of output classes for the final logits.
          dropout_rate: the dropout rate applied to the head projection.
          conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
              ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
              own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
              uses two sequential 3D ops instead.
          activation: the input activation name.
          output_activation: optional final activation (e.g., 'softmax').
          max_pool_predictions: apply temporal softmax pooling to predictions.
              Intended for multi-label prediction, where multiple labels are
              distributed across the video. Currently only supports single clips.
          kernel_initializer: kernel initializer for the conv operations.
          kernel_regularizer: kernel regularizer for the conv operations.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(ClassifierHead, self).__init__(**kwargs)

        self._head_filters = head_filters
        self._num_classes = num_classes
        self._dropout_rate = dropout_rate
        self._conv_type = conv_type
        self._activation = activation
        self._output_activation = output_activation
        self._max_pool_predictions = max_pool_predictions
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer

        self._dropout = keras.layers.Dropout(dropout_rate)
        self._head = ConvBlock(
            filters=head_filters,
            kernel_size=1,
            activation=activation,
            use_bias=True,
            use_batch_norm=False,
            conv_type=conv_type,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="head",
        )
        self._classifier = ConvBlock(
            filters=num_classes,
            kernel_size=1,
            kernel_initializer=keras.initializers.random_normal(stddev=0.01),
            kernel_regularizer=None,
            use_bias=True,
            use_batch_norm=False,
            conv_type=conv_type,
            name="classifier",
        )
        self._max_pool = TemporalSoftmaxPool()
        self._squeeze = Squeeze3D()

        output_activation = output_activation if output_activation else "linear"
        self._cast = keras.layers.Activation(
            output_activation, dtype="float32", name="cast"
        )

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "head_filters": self._head_filters,
            "num_classes": self._num_classes,
            "dropout_rate": self._dropout_rate,
            "conv_type": self._conv_type,
            "activation": self._activation,
            "output_activation": self._output_activation,
            "max_pool_predictions": self._max_pool_predictions,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
        }
        base_config = super(ClassifierHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the layer with the given inputs."""
        # Input Shape: [batch_size, 1, 1, 1, input_channels]
        x = inputs

        x = self._head(x)

        if self._dropout_rate and self._dropout_rate > 0:
            x = self._dropout(x)

        x = self._classifier(x)

        if self._max_pool_predictions:
            x = self._max_pool(x)

        x = self._squeeze(x)
        x = self._cast(x)

        return x