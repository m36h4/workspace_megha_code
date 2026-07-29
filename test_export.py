import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# Copy ONLY your PrimitiveHSwish implementation here
class PrimitiveHSwish(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # paste your implementation here
        pass

class Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.act = PrimitiveHSwish()

    def forward(self, x):
        return self.act(x)

model = Model()
model.eval()

spec = [paddle.static.InputSpec([1, 16, 32, 32], dtype="float32")]

paddle.jit.save(
    paddle.jit.to_static(model, input_spec=spec),
    "test_model"
)

print("Export successful")
