
from typing import Tuple, Mapping, Sequence, Callable, Dict, Union
import tensorflow as tf
import dataclasses


# Defines a set of kernel sizes and stride sizes to simplify and shorten
# architecture definitions for configs below.
KernelSize = Tuple[int, int, int]
Activation = Union[str, Callable]
States = Dict[str, tf.Tensor]

# K(ab) represents a 3D kernel of size (a, b, b)
K13: KernelSize = (1, 3, 3)
K15: KernelSize = (1, 5, 5)
K33: KernelSize = (3, 3, 3)
K53: KernelSize = (5, 3, 3)

# S(ab) represents a 3D stride of size (a, b, b)
S11: KernelSize = (1, 1, 1)
S12: KernelSize = (1, 2, 2)
S22: KernelSize = (2, 2, 2)
S21: KernelSize = (2, 1, 1)

# Type for a state container (map)
TensorMap = Mapping[str, tf.Tensor]


@dataclasses.dataclass
class BlockSpec:
    """Configuration of a block."""


@dataclasses.dataclass
class StemSpec(BlockSpec):
    """Configuration of a Movinet block."""

    filters: int = 0
    kernel_size: KernelSize = (0, 0, 0)
    strides: KernelSize = (0, 0, 0)


@dataclasses.dataclass
class MovinetBlockSpec(BlockSpec):
    """Configuration of a Movinet block."""

    base_filters: int = 0
    expand_filters: Sequence[int] = ()
    kernel_sizes: Sequence[KernelSize] = ()
    strides: Sequence[KernelSize] = ()


@dataclasses.dataclass
class HeadSpec(BlockSpec):
    """Configuration of a Movinet block."""

    project_filters: int = 0
    head_filters: int = 0


# Block specs specify the architecture of each model
BLOCK_SPECS = {
    "a0": (
        StemSpec(filters=8, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=8, expand_filters=(24,), kernel_sizes=(K15,), strides=(S12,)
        ),
        MovinetBlockSpec(
            base_filters=32,
            expand_filters=(80, 80, 80),
            kernel_sizes=(K33, K33, K33),
            strides=(S12, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(184, 112, 184),
            kernel_sizes=(K53, K33, K33),
            strides=(S12, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(184, 184, 184, 184),
            kernel_sizes=(K53, K33, K33, K33),
            strides=(S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=104,
            expand_filters=(384, 280, 280, 344),
            kernel_sizes=(K53, K15, K15, K15),
            strides=(S12, S11, S11, S11),
        ),
        HeadSpec(project_filters=480, head_filters=2048),
    ),
    "a1": (
        StemSpec(filters=16, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=16,
            expand_filters=(40, 40),
            kernel_sizes=(K15, K33),
            strides=(S12, S11),
        ),
        MovinetBlockSpec(
            base_filters=40,
            expand_filters=(96, 120, 96, 96),
            kernel_sizes=(K33, K33, K33, K33),
            strides=(S12, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=64,
            expand_filters=(216, 128, 216, 168, 216),
            kernel_sizes=(K53, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=64,
            expand_filters=(216, 216, 216, 128, 128, 216),
            kernel_sizes=(K53, K33, K33, K33, K15, K33),
            strides=(S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=136,
            expand_filters=(456, 360, 360, 360, 456, 456, 544),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K13),
            strides=(S12, S11, S11, S11, S11, S11, S11),
        ),
        HeadSpec(project_filters=600, head_filters=2048),
    ),
    "a2": (
        StemSpec(filters=16, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=16,
            expand_filters=(40, 40, 64),
            kernel_sizes=(K15, K33, K33),
            strides=(S12, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=40,
            expand_filters=(96, 120, 96, 96, 120),
            kernel_sizes=(K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=72,
            expand_filters=(240, 160, 240, 192, 240),
            kernel_sizes=(K53, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=72,
            expand_filters=(240, 240, 240, 240, 144, 240),
            kernel_sizes=(K53, K33, K33, K33, K15, K33),
            strides=(S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=144,
            expand_filters=(480, 384, 384, 480, 480, 480, 576),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K13),
            strides=(S12, S11, S11, S11, S11, S11, S11),
        ),
        HeadSpec(project_filters=640, head_filters=2048),
    ),
    "a3": (
        StemSpec(filters=16, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=16,
            expand_filters=(40, 40, 64, 40),
            kernel_sizes=(K15, K33, K33, K33),
            strides=(S12, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=48,
            expand_filters=(112, 144, 112, 112, 144, 144),
            kernel_sizes=(K33, K33, K33, K15, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=80,
            expand_filters=(240, 152, 240, 192, 240),
            kernel_sizes=(K53, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=88,
            expand_filters=(264, 264, 264, 264, 160, 264, 264, 264),
            kernel_sizes=(K53, K33, K33, K33, K15, K33, K33, K33),
            strides=(S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=168,
            expand_filters=(560, 448, 448, 560, 560, 560, 448, 448, 560, 672),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K15, K15, K33, K13),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        HeadSpec(project_filters=744, head_filters=2048),
    ),
    "a4": (
        StemSpec(filters=24, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=24,
            expand_filters=(64, 64, 96, 64, 96, 64),
            kernel_sizes=(K15, K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(168, 168, 136, 136, 168, 168, 168, 136, 136),
            kernel_sizes=(K33, K33, K33, K33, K33, K33, K33, K15, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=96,
            expand_filters=(320, 160, 320, 192, 320, 160, 320, 256, 320),
            kernel_sizes=(K53, K33, K33, K33, K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=96,
            expand_filters=(320, 320, 320, 320, 192, 320, 320, 192, 320, 320),
            kernel_sizes=(K53, K33, K33, K33, K15, K33, K33, K33, K33, K33),
            strides=(S11, S11, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=192,
            expand_filters=(
                640,
                512,
                512,
                640,
                640,
                640,
                512,
                512,
                640,
                768,
                640,
                640,
                768,
            ),
            kernel_sizes=(
                K53,
                K15,
                K15,
                K15,
                K15,
                K33,
                K15,
                K15,
                K15,
                K15,
                K15,
                K33,
                K33,
            ),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        HeadSpec(project_filters=856, head_filters=2048),
    ),
    "a5": (
        StemSpec(filters=24, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=24,
            expand_filters=(64, 64, 96, 64, 96, 64),
            kernel_sizes=(K15, K15, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=64,
            expand_filters=(192, 152, 152, 152, 192, 192, 192, 152, 152, 192, 192),
            kernel_sizes=(K53, K33, K33, K33, K33, K33, K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=112,
            expand_filters=(
                376,
                224,
                376,
                376,
                296,
                376,
                224,
                376,
                376,
                296,
                376,
                376,
                376,
            ),
            kernel_sizes=(
                K53,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
                K33,
            ),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=120,
            expand_filters=(376, 376, 376, 376, 224, 376, 376, 224, 376, 376, 376),
            kernel_sizes=(K53, K33, K33, K33, K15, K33, K33, K33, K33, K33, K33),
            strides=(S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=224,
            expand_filters=(
                744,
                744,
                600,
                600,
                744,
                744,
                744,
                896,
                600,
                600,
                896,
                744,
                744,
                896,
                600,
                600,
                744,
                744,
            ),
            kernel_sizes=(
                K53,
                K33,
                K15,
                K15,
                K15,
                K15,
                K33,
                K15,
                K15,
                K15,
                K15,
                K15,
                K33,
                K15,
                K15,
                K15,
                K15,
                K33,
            ),
            strides=(
                S12,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
                S11,
            ),
        ),
        HeadSpec(project_filters=992, head_filters=2048),
    ),
    "t0": (
        StemSpec(filters=8, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=8, expand_filters=(16,), kernel_sizes=(K15,), strides=(S12,)
        ),
        MovinetBlockSpec(
            base_filters=32,
            expand_filters=(72, 72),
            kernel_sizes=(K33, K15),
            strides=(S12, S11),
        ),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(112, 112, 112),
            kernel_sizes=(K53, K15, K33),
            strides=(S12, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(184, 184, 184, 184),
            kernel_sizes=(K53, K15, K33, K33),
            strides=(S11, S11, S11, S11),
        ),
        MovinetBlockSpec(
            base_filters=104,
            expand_filters=(344, 344, 344, 344),
            kernel_sizes=(K53, K15, K15, K33),
            strides=(S12, S11, S11, S11),
        ),
        HeadSpec(project_filters=240, head_filters=1024),
    ),
}