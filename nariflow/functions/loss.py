import os
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../conf.txt")
with open(path, mode = 'r') as f:
    conf = f.readline()
    if conf == 'node':
        from ..core.core import Variable, Function, CalcGradient
    if conf == 'tape':
        from ..core.core_tape import Variable, Function, GradientTape
import numpy as np
from ..core.elementary_function import exp, log, flowsum


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0 = self.input_list[0]
        x1 = self.input_list[1]
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff.data))
        gx1 = -gx0
        return gx0, gx1

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def categorical_crossentropy(real, pred):
    y = - flowsum(real * log(pred + 1e-15))
    y =   y / pred.data.shape[0]
    return y