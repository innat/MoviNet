import tenosrflow as tf 
from tensorflow import keras
from typing import Optional, Tuple
from movinet.layers.scale import Scale
from movinet.layers.stochastic_depth import StochasticDepth
from movinet.config import States


class MobileBottleneck(keras.layers.Layer):
    """A depthwise inverted bottleneck block.

    Uses dependency injection to allow flexible definition of different layers
    within this block.
    """

    def __init__(
        self,
        expansion_layer: keras.layers.Layer,
        feature_layer: keras.layers.Layer,
        projection_layer: keras.layers.Layer,
        attention_layer: Optional[keras.layers.Layer] = None,
        skip_layer: Optional[keras.layers.Layer] = None,
        stochastic_depth_drop_rate: Optional[float] = None,
        **kwargs
    ):
        """Implementation for mobile bottleneck.

        Args:
          expansion_layer: initial layer used for pointwise expansion.
          feature_layer: main layer used for computing 3D features.
          projection_layer: layer used for pointwise projection.
          attention_layer: optional layer used for attention-like operations (e.g.,
              squeeze excite).
          skip_layer: optional skip layer used to project the input before summing
              with the output for the residual connection.
          stochastic_depth_drop_rate: optional drop rate for stochastic depth.
          **kwargs: keyword arguments to be passed to this layer.
        """
        super(MobileBottleneck, self).__init__(**kwargs)

        self._projection_layer = projection_layer
        self._attention_layer = attention_layer
        self._skip_layer = skip_layer
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._identity = keras.layers.Activation(tf.identity)
        self._rezero = Scale(initializer="zeros", name="rezero")

        if stochastic_depth_drop_rate:
            self._stochastic_depth = StochasticDepth(
                stochastic_depth_drop_rate, name="stochastic_depth"
            )
        else:
            self._stochastic_depth = None

        self._feature_layer = feature_layer
        self._expansion_layer = expansion_layer

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "stochastic_depth_drop_rate": self._stochastic_depth_drop_rate,
        }
        base_config = super(MobileBottleneck, self).get_config()
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

        x = self._expansion_layer(inputs)
        x, states = self._feature_layer(x, states=states)
        if self._attention_layer is not None:
            x, states = self._attention_layer(x, states=states)
        x = self._projection_layer(x)

        # Add identity so that the ops are ordered as written. This is useful for,
        # e.g., quantization.
        x = self._identity(x)
        x = self._rezero(x)

        if self._stochastic_depth is not None:
            x = self._stochastic_depth(x)

        if self._skip_layer is not None:
            skip = self._skip_layer(inputs)
        else:
            skip = inputs

        return x + skip, states