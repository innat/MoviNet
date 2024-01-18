
from typing import List
import tensorflow as tf
from tensorflow import keras

class CausalConvMixin:
    """Mixin class to implement CausalConv for `keras.layers.Conv` layers."""

    @property
    def use_buffered_input(self) -> bool:
        return self._use_buffered_input

    @use_buffered_input.setter
    def use_buffered_input(self, variable: bool):
        self._use_buffered_input = variable

    def _compute_buffered_causal_padding(
        self,
        inputs: tf.Tensor,
        use_buffered_input: bool = False,
        time_axis: int = 1,
    ) -> List[List[int]]:
        """Calculates padding for 'causal' option for conv layers.

        Args:
          inputs: An optional input `tf.Tensor` to be padded.
          use_buffered_input: A `bool`. If True, use 'valid' padding along the time
            dimension. This should be set when applying the stream buffer.
          time_axis: An `int` of the axis of the time dimension.

        Returns:
          A list of paddings for `tf.pad`.
        """
        input_shape = tf.shape(inputs)[1:-1]

        if keras.backend.image_data_format() == "channels_first":
            raise ValueError('"channels_first" mode is unsupported.')

        kernel_size_effective = [
            (
                self.kernel_size[i]
                + (self.kernel_size[i] - 1) * (self.dilation_rate[i] - 1)
            )
            for i in range(self.rank)
        ]
        pad_total = [kernel_size_effective[0] - 1]
        for i in range(1, self.rank):
            overlap = (input_shape[i] - 1) % self.strides[i] + 1
            pad_total.append(tf.maximum(kernel_size_effective[i] - overlap, 0))
        pad_beg = [pad_total[i] // 2 for i in range(self.rank)]
        pad_end = [pad_total[i] - pad_beg[i] for i in range(self.rank)]
        padding = [[pad_beg[i], pad_end[i]] for i in range(self.rank)]
        padding = [[0, 0]] + padding + [[0, 0]]

        if use_buffered_input:
            padding[time_axis] = [0, 0]
        else:
            padding[time_axis] = [padding[time_axis][0] + padding[time_axis][1], 0]
        return padding

    def _causal_validate_init(self):
        """Validates the Conv layer initial configuration."""
        # Overriding this method is meant to circumvent unnecessary errors when
        # using causal padding.
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                "The number of filters must be evenly divisible by the number of "
                "groups. Received: groups={}, filters={}".format(
                    self.groups, self.filters
                )
            )

        if not all(self.kernel_size):
            raise ValueError(
                "The argument `kernel_size` cannot contain 0(s). "
                "Received: %s" % (self.kernel_size,)
            )

    def _buffered_spatial_output_shape(self, spatial_output_shape: List[int]):
        """Computes the spatial output shape from the input shape."""
        # When buffer padding, use 'valid' padding across time. The output shape
        # across time should be the input shape minus any padding, assuming
        # the stride across time is 1.
        if self._use_buffered_input and spatial_output_shape[0] is not None:
            padding = self._compute_buffered_causal_padding(
                tf.zeros([1] + spatial_output_shape + [1]), use_buffered_input=False
            )
            spatial_output_shape[0] -= sum(padding[1])
        return spatial_output_shape