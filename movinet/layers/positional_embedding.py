
import tenosrflow as tf 
from tensorflow import keras
from typing import Union, Optional, Tuple
from movinet.config import States

class PositionalEncoding(keras.layers.Layer):
    """Creates a network layer that adds a sinusoidal positional encoding.

    Positional encoding is incremented across frames, and is added to the input.
    The positional encoding is first weighted at 0 so that the network can choose
    to ignore it. This implements:

    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
    Attention Is All You Need.
    (https://arxiv.org/pdf/1706.03762.pdf).
    """

    def __init__(
        self,
        initializer: keras.initializers.Initializer = "zeros",
        cache_encoding: bool = False,
        state_prefix: Optional[str] = None,
        **kwargs,
    ):
        """Initializes positional encoding.

        Args:
          initializer: A `str` of initializer for weighting the positional encoding.
          cache_encoding: A `bool`. If True, cache the positional encoding tensor
            after calling build. Otherwise, rebuild the tensor for every call.
            Setting this to False can be useful when we want to input a variable
            number of frames, so the positional encoding tensor can change shape.
          state_prefix: a prefix string to identify states.
          **kwargs: Additional keyword arguments to be passed to this layer.

        Returns:
          A `tf.Tensor` of which should have the same shape as input.
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self._initializer = initializer
        self._cache_encoding = cache_encoding
        self._pos_encoding = None
        self._rezero = Scale(initializer=initializer, name="rezero")
        state_prefix = state_prefix if state_prefix is not None else ""
        self._state_prefix = state_prefix
        self._frame_count_name = f"{state_prefix}_pos_enc_frame_count"

    def get_config(self):
        """Returns a dictionary containing the config used for initialization."""
        config = {
            "initializer": self._initializer,
            "cache_encoding": self._cache_encoding,
            "state_prefix": self._state_prefix,
        }
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _positional_encoding(
        self,
        num_positions: Union[int, tf.Tensor],
        hidden_size: Union[int, tf.Tensor],
        start_position: Union[int, tf.Tensor] = 0,
        dtype: str = "float32",
    ) -> tf.Tensor:
        """Creates a sequence of sinusoidal positional encoding vectors.

        Args:
          num_positions: the total number of positions (frames).
          hidden_size: the number of channels used for the hidden vectors.
          start_position: the start position.
          dtype: the dtype of the output tensor.

        Returns:
          The positional encoding tensor with shape [num_positions, hidden_size].
        """
        if isinstance(start_position, tf.Tensor) and start_position.shape.rank == 1:
            start_position = start_position[0]

        # Calling `tf.range` with `dtype=tf.bfloat16` results in an error,
        # so we cast afterward.
        positions = tf.range(start_position, start_position + num_positions)
        positions = tf.cast(positions, dtype)[:, tf.newaxis]
        idx = tf.range(hidden_size)[tf.newaxis, :]

        power = tf.cast(2 * (idx // 2), dtype)
        power /= tf.cast(hidden_size, dtype)
        angles = 1.0 / tf.math.pow(10_000.0, power)
        radians = positions * angles

        sin = tf.math.sin(radians[:, 0::2])
        cos = tf.math.cos(radians[:, 1::2])
        pos_encoding = tf.concat([sin, cos], axis=-1)

        return pos_encoding

    def _get_pos_encoding(
        self, input_shape: tf.Tensor, frame_count: int = 0
    ) -> tf.Tensor:
        """Calculates the positional encoding from the input shape.

        Args:
          input_shape: the shape of the input.
          frame_count: a count of frames that indicates the index of the first
            frame.

        Returns:
          The positional encoding tensor with shape [num_positions, hidden_size].

        """
        frames = input_shape[1]
        channels = input_shape[-1]
        pos_encoding = self._positional_encoding(
            frames, channels, start_position=frame_count, dtype=self.dtype
        )
        pos_encoding = tf.reshape(pos_encoding, [1, frames, 1, 1, channels])
        return pos_encoding

    def build(self, input_shape):
        """Builds the layer with the given input shape.

        Args:
          input_shape: The input shape.

        Raises:
          ValueError: If using 'channels_first' data format.
        """
        if tf.keras.backend.image_data_format() == "channels_first":
            raise ValueError('"channels_first" mode is unsupported.')

        if self._cache_encoding:
            self._pos_encoding = self._get_pos_encoding(input_shape)

        super(PositionalEncoding, self).build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        states: Optional[States] = None,
        output_states: bool = True,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, States]]:
        """Calls the layer with the given inputs.

        Args:
          inputs: An input `tf.Tensor`.
          states: A `dict` of states such that, if any of the keys match for this
            layer, will overwrite the contents of the buffer(s). Expected keys
            include `state_prefix + '_pos_enc_frame_count'`.
          output_states: A `bool`. If True, returns the output tensor and output
            states. Returns just the output tensor otherwise.

        Returns:
          An output `tf.Tensor` (and optionally the states if `output_states=True`).

        Raises:
          ValueError: If using 'channels_first' data format.
        """
        states = dict(states) if states is not None else {}

        # Keep a count of frames encountered across input iterations in
        # num_frames to be able to accurately update the positional encoding.
        num_frames = tf.shape(inputs)[1]
        frame_count = tf.cast(states.get(self._frame_count_name, [0]), tf.int32)
        states[self._frame_count_name] = frame_count + num_frames

        if self._cache_encoding:
            pos_encoding = self._pos_encoding
        else:
            pos_encoding = self._get_pos_encoding(
                tf.shape(inputs), frame_count=frame_count
            )
        pos_encoding = tf.cast(pos_encoding, inputs.dtype)
        pos_encoding = self._rezero(pos_encoding)
        outputs = inputs + pos_encoding

        return (outputs, states) if output_states else outputs