
import tensorflow as tf 
from tensorflow import keras

from typing import Optional, Any, Tuple
from movinet.config import States

class StreamBuffer(keras.layers.Layer):
    """Stream buffer wrapper which caches activations of previous frames."""

    def __init__(self, buffer_size: int, state_prefix: Optional[str] = None, **kwargs):
        """Initializes a stream buffer.

        Args:
          buffer_size: the number of input frames to cache.
          state_prefix: a prefix string to identify states.
          **kwargs: keyword arguments to be passed to this layer.

        Returns:
          A output tensor of the StreamBuffer operation.
        """
        super(StreamBuffer, self).__init__(**kwargs)

        state_prefix = state_prefix if state_prefix is not None else ""
        self._state_prefix = state_prefix
        self._state_name = f"{state_prefix}_stream_buffer"
        self._buffer_size = buffer_size

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "buffer_size": self._buffer_size,
            "state_prefix": self._state_prefix,
        }
        base_config = super(StreamBuffer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(
        self,
        inputs: tf.Tensor,
        states: Optional[States] = None,
    ) -> Tuple[Any, States]:
        """Calls the layer with the given inputs.

        Args:
          inputs: the input tensor.
          states: a dict of states such that, if any of the keys match for this
              layer, will overwrite the contents of the buffer(s).
              Expected keys include `state_prefix + '_stream_buffer'`.

        Returns:
          the output tensor and states
        """
        states = dict(states) if states is not None else {}
        buffer = states.get(self._state_name, None)

        # Create the buffer if it does not exist in the states.
        # Output buffer shape:
        # [batch_size, buffer_size, input_height, input_width, num_channels]
        if buffer is None:
            shape = tf.shape(inputs)
            buffer = tf.zeros(
                [shape[0], self._buffer_size, shape[2], shape[3], shape[4]],
                dtype=inputs.dtype,
            )

        # tf.pad has limited support for tf lite, so use tf.concat instead.
        full_inputs = tf.concat([buffer, inputs], axis=1)

        # Cache the last b frames of the input where b is the buffer size and f
        # is the number of input frames. If b > f, then we will cache the last b - f
        # frames from the previous buffer concatenated with the current f input
        # frames.
        new_buffer = full_inputs[:, -self._buffer_size :]
        states[self._state_name] = new_buffer

        return full_inputs, states