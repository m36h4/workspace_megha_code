# custom_activation.py
#
# Primitive implementations of HardSigmoid and HardSwish
# for ONNX export compatibility.
#
# PrimitiveHSigmoid:
#     clip(x + 3, 0, 6) / 6
#
# PrimitiveHSwish:
#     x * clip(x + 3, 0, 6) / 6

import paddle
import paddle.nn as nn

__all__ = [
    "PrimitiveHSigmoid",
    "PrimitiveHSwish",
]


class PrimitiveHSigmoid(nn.Layer):
    """
    Primitive implementation of HardSigmoid using only
    ONNX-friendly operators.

    ONNX:
        Add -> Clip -> Div
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return paddle.clip(
            x + 3.0,
            min=0.0,
            max=6.0
        ) / 6.0


class PrimitiveHSwish(nn.Layer):
    """
    Primitive implementation of HardSwish using only
    ONNX-friendly operators.

    ONNX:
        Add -> Clip -> Div -> Mul
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (
            paddle.clip(
                x + 3.0,
                min=0.0,
                max=6.0
            ) / 6.0
        )
