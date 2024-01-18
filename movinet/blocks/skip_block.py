
import tensorflow as tf 
from tensorflow import keras
from typing import Optional
from movinet.blocks.conv import ConvBlock

KERNEL_WEIGHT_DECAY = 1.5e-5

class SkipBlock(keras.layers.Layer):
    """Skip block for bottleneck blocks."""

    def __init__(
        self,
        out_filters: int,
        downsample: bool = False,
        conv_type: str = "3d",
        kernel_initializer: keras.initializers.Initializer = "HeNormal",
        kernel_regularizer: Optional[
            keras.regularizers.Regularizer
        ] = keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
        batch_norm_layer: keras.layers.Layer = keras.layers.BatchNormalization,
        batch_norm_momentum: float = 0.99,
        batch_norm_epsilon: float = 1e-3,  # pytype: disable=annotation-type-mismatch  # typed-keras
        use_sync_bn: bool = False,
        **kwargs
    ):
        """Implementation for skip block.

        Args:
          out_filters: the number of projected output filters.
          downsample: if True, downsamples the input by a factor of 2 by applying
              average pooling with a 3x3 kernel size on the spatial dimensions.
          conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
              ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
              own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
              uses two sequential 3D ops instead.
          kernel_initializer: kernel initializer for the conv operations.
          kernel_regularizer: kernel regularizer for the conv projection.
          batch_norm_layer: class to use for batch norm.
          batch_norm_momentum: momentum of the batch norm operation.
          batch_norm_epsilon: epsilon of the batch norm operation.
          use_sync_bn: if True, use synchronized batch normalization.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(SkipBlock, self).__init__(**kwargs)

        self._out_filters = out_filters
        self._downsample = downsample
        self._conv_type = conv_type
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._batch_norm_layer = batch_norm_layer
        self._batch_norm_momentum = batch_norm_momentum
        self._batch_norm_epsilon = batch_norm_epsilon
        self._use_sync_bn = use_sync_bn

        self._projection = ConvBlock(
            filters=self._out_filters,
            kernel_size=1,
            conv_type=conv_type,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_batch_norm=True,
            batch_norm_layer=self._batch_norm_layer,
            batch_norm_momentum=self._batch_norm_momentum,
            batch_norm_epsilon=self._batch_norm_epsilon,
            use_sync_bn=self._use_sync_bn,
            name="skip_project",
        )

        if downsample:
            if self._conv_type == "2plus1d":
                self._pool = keras.layers.AveragePooling2D(
                    pool_size=(3, 3), strides=(2, 2), padding="same", name="skip_pool"
                )
            else:
                self._pool = keras.layers.AveragePooling3D(
                    pool_size=(1, 3, 3),
                    strides=(1, 2, 2),
                    padding="same",
                    name="skip_pool",
                )
        else:
            self._pool = None

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "out_filters": self._out_filters,
            "downsample": self._downsample,
            "conv_type": self._conv_type,
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "batch_norm_momentum": self._batch_norm_momentum,
            "batch_norm_epsilon": self._batch_norm_epsilon,
            "use_sync_bn": self._use_sync_bn,
        }
        base_config = super(SkipBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Calls the layer with the given inputs."""
        x = inputs
        if self._pool is not None:
            if self._conv_type == "2plus1d":
                x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3], x.shape[4]])

            x = self._pool(x)

            if self._conv_type == "2plus1d":
                x = tf.reshape(
                    x,
                    [
                        tf.shape(inputs)[0],
                        -1,
                        tf.shape(x)[1],
                        tf.shape(x)[2],
                        x.shape[3],
                    ],
                )
        return self._projection(x)