
from typing import Union, Tuple, Optional
import tensorflow as tf 
from tensorflow import keras
import functools, six

def make_divisible(
    value: float,
    divisor: int,
    min_value: Optional[float] = None,
    round_down_protect: bool = True,
) -> int:
    """This is to ensure that all layers have channels that are divisible by 8.

    Args:
      value: A `float` of original value.
      divisor: An `int` of the divisor that need to be checked upon.
      min_value: A `float` of  minimum value threshold.
      round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
      The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def normalize_tuple(value: Union[int, Tuple[int, ...]], size: int, name: str):
    """Transforms a single integer or iterable of integers into an integer tuple.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable of
        ints.
      size: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.
    Returns:
      A tuple of `size` integers.
    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * size
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(size)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != size:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(size)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(size)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple
    


def get_activation(identifier, use_keras_layer=False, **kwargs):
    """Maps an identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf_keras.activations.get.

    Prefers using keras layers when use_keras_layer=True. Now it only supports
    'relu', 'linear', 'identity', 'swish', 'mish', 'leaky_relu', and 'gelu'.

    Args:
      identifier: String name of the activation function or callable.
      use_keras_layer: If True, use keras layer if identifier is allow-listed.
      **kwargs: Keyword arguments to use to instantiate an activation function.
        Available only for 'leaky_relu' and 'gelu' when using keras layers.
        For example: get_activation('leaky_relu', use_keras_layer=True, alpha=0.1)

    Returns:
      A Python function corresponding to the activation function or a keras
      activation layer when use_keras_layer=True.
    """
    if isinstance(identifier, six.string_types):
        identifier = str(identifier).lower()
        if use_keras_layer:
            keras_layer_allowlist = {
                "relu": "relu",
                "linear": "linear",
                "identity": "linear",
                "swish": "swish",
                "sigmoid": "sigmoid",
                "relu6": tf.nn.relu6,
                "leaky_relu": functools.partial(tf.nn.leaky_relu, **kwargs),
                "hard_swish": hard_swish,
                "hard_sigmoid": hard_sigmoid,
                "mish": mish,
                "gelu": functools.partial(tf.nn.gelu, **kwargs),
            }
            if identifier in keras_layer_allowlist:
                return keras.layers.Activation(keras_layer_allowlist[identifier])
        name_to_fn = {
            "gelu": gelu,
            "simple_swish": simple_swish,
            "hard_swish": hard_swish,
            "relu6": relu6,
            "hard_sigmoid": hard_sigmoid,
            "identity": identity,
            "mish": mish,
        }
        if identifier in name_to_fn:
            return keras.activations.get(name_to_fn[identifier])
    return keras.activations.get(identifier)


@tf.keras.utils.register_keras_serializable(package="Text")
def hard_swish(features):
    """Computes a hard version of the swish function.

    This operation can be used to reduce computational cost and improve
    quantization for edge devices.

    Args:
      features: A `Tensor` representing preactivation values.

    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    fdtype = features.dtype
    return features * tf.nn.relu6(features + tf.cast(3.0, fdtype)) * (1.0 / 6.0)


@tf.keras.utils.register_keras_serializable(package="Text")
def hard_sigmoid(features):
    """Computes the hard sigmoid activation function.

    Args:
      features: A `Tensor` representing preactivation values.

    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return tf.nn.relu6(features + tf.cast(3.0, features.dtype)) * 0.16667


@tf.keras.utils.register_keras_serializable(package="Text")
def mish(x) -> tf.Tensor:
    """Mish activation function.

       Mish: A Self Regularized Non-Monotonic Activation Function
       https://arxiv.org/pdf/1908.08681.pdf

       Mish(x) = x * tanh(ln(1+e^x))

    Args:
      x: A `Tensor` representing preactivation values.

    Returns:
      The activation value.
    """
    x = tf.convert_to_tensor(x)
    return x * tf.tanh(tf.nn.softplus(x))


@tf.keras.utils.register_keras_serializable(package="Text")
def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.

    Returns:
      `x` with the GELU activation applied.
    """
    return tf.keras.activations.gelu(x, approximate=True)


@tf.keras.utils.register_keras_serializable(package="Text")
def simple_swish(features):
    """Computes the Swish activation function.

    The tf.nn.swish operation uses a custom gradient to reduce memory usage.
    Since saving custom gradients in SavedModel is currently not supported, and
    one would not be able to use an exported TF-Hub module for fine-tuning, we
    provide this wrapper that can allow to select whether to use the native
    TensorFlow swish operation, or whether to use a customized operation that
    has uses default TensorFlow gradient computation.

    Args:
      features: A `Tensor` representing preactivation values.

    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return features * tf.nn.sigmoid(features)


@tf.keras.utils.register_keras_serializable(package="Text")
def relu6(features):
    """Computes the Relu6 activation function.

    Args:
      features: A `Tensor` representing preactivation values.

    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return tf.nn.relu6(features)


@tf.keras.utils.register_keras_serializable(package="Text")
def identity(features):
    """Computes the identity function.

    Useful for helping in quantization.

    Args:
      features: A `Tensor` representing preactivation values.

    Returns:
      The activation value.
    """
    features = tf.convert_to_tensor(features)
    return tf.identity(features)