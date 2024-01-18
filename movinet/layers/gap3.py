
import tensorflow as tf
from tensorflow import keras
from typing import Union, Optional, Tuple
from movinet.config import States

class GlobalAveragePool3D(keras.layers.Layer):
    """Creates a global average pooling layer with causal mode.

    Implements causal mode, which runs a cumulative sum (with `tf.cumsum`) across
    frames in the time dimension, allowing the use of a stream buffer. Sums any
    valid input state with the current input to allow state to accumulate over
    several iterations.
    """

    def __init__(
        self,
        keepdims: bool = False,
        causal: bool = False,
        state_prefix: Optional[str] = None,
        **kwargs,
    ):
        """Initializes a global average pool layer.

        Args:
          keepdims: A `bool`. If True, keep the averaged dimensions.
          causal: A `bool` of whether to run in causal mode with a cumulative sum
            across frames.
          state_prefix: a prefix string to identify states.
          **kwargs: Additional keyword arguments to be passed to this layer.

        Returns:
          An output `tf.Tensor`.
        """
        super(GlobalAveragePool3D, self).__init__(**kwargs)

        self._keepdims = keepdims
        self._causal = causal
        state_prefix = state_prefix if state_prefix is not None else ""
        self._state_prefix = state_prefix

        self._state_name = f"{state_prefix}_pool_buffer"
        self._frame_count_name = f"{state_prefix}_pool_frame_count"

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "keepdims": self._keepdims,
            "causal": self._causal,
            "state_prefix": self._state_prefix,
        }
        base_config = super(GlobalAveragePool3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(
        self,
        inputs: tf.Tensor,
        states: Optional[States] = None,
        output_states: bool = False,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, States]]:
        """Calls the layer with the given inputs.

        Args:
          inputs: An input `tf.Tensor`.
          states: A `dict` of states such that, if any of the keys match for this
            layer, will overwrite the contents of the buffer(s).
            Expected keys include `state_prefix + '__pool_buffer'` and
            `state_prefix + '__pool_frame_count'`.
          output_states: A `bool`. If True, returns the output tensor and output
            states. Returns just the output tensor otherwise.

        Returns:
          An output `tf.Tensor` (and optionally the states if `output_states=True`).
          If `causal=True`, the output tensor will have shape
          `[batch_size, num_frames, 1, 1, channels]` if `keepdims=True`. We keep
          the frame dimension in this case to simulate a cumulative global average
          as if we are inputting one frame at a time. If `causal=False`, the output
          is equivalent to `tf_keras.layers.GlobalAveragePooling3D` with shape
          `[batch_size, 1, 1, 1, channels]` if `keepdims=True` (plus the optional
          buffer stored in `states`).

        Raises:
          ValueError: If using 'channels_first' data format.
        """
        states = dict(states) if states is not None else {}

        if tf.keras.backend.image_data_format() == "channels_first":
            raise ValueError('"channels_first" mode is unsupported.')

        # Shape: [batch_size, 1, 1, 1, channels]
        buffer = states.get(self._state_name, None)
        if buffer is None:
            buffer = tf.zeros_like(inputs[:, :1, :1, :1], dtype=inputs.dtype)
            states[self._state_name] = buffer

        # Keep a count of frames encountered across input iterations in
        # num_frames to be able to accurately take a cumulative average across
        # all frames when running in streaming mode
        num_frames = tf.shape(inputs)[1]
        frame_count = states.get(self._frame_count_name, tf.constant([0]))
        frame_count = tf.cast(frame_count, tf.int32)
        states[self._frame_count_name] = frame_count + num_frames

        if self._causal:
            # Take a mean of spatial dimensions to make computation more efficient.
            x = tf.reduce_mean(inputs, axis=[2, 3], keepdims=True)
            x = tf.cumsum(x, axis=1)
            x = x + buffer

            # The last frame will be the value of the next state
            # Shape: [batch_size, 1, 1, 1, channels]
            states[self._state_name] = x[:, -1:]

            # In causal mode, the divisor increments by 1 for every frame to
            # calculate cumulative averages instead of one global average
            mean_divisors = tf.range(num_frames) + frame_count + 1
            mean_divisors = tf.reshape(mean_divisors, [1, num_frames, 1, 1, 1])
            mean_divisors = tf.cast(mean_divisors, x.dtype)

            # Shape: [batch_size, num_frames, 1, 1, channels]
            x = x / mean_divisors
        else:
            # In non-causal mode, we (optionally) sum across frames to take a
            # cumulative average across input iterations rather than individual
            # frames. If no buffer state is passed, this essentially becomes
            # regular global average pooling.
            # Shape: [batch_size, 1, 1, 1, channels]
            x = tf.reduce_sum(inputs, axis=(1, 2, 3), keepdims=True)
            x = x / tf.cast(tf.shape(inputs)[2] * tf.shape(inputs)[3], x.dtype)
            x = x + buffer

            # Shape: [batch_size, 1, 1, 1, channels]
            states[self._state_name] = x

            x = x / tf.cast(frame_count + num_frames, x.dtype)

        if not self._keepdims:
            x = tf.squeeze(x, axis=(1, 2, 3))

        return (x, states) if output_states else x