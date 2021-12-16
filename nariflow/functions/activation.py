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
from ..core.elementary_function import *

def sigmoid(x):
    y = 1 / (1 + exp(-x))
    return y

class Relu(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return Relu()(x)

def softmax(x, axis = 1):
    ex = exp(x)
    sum_ex = flowsum(ex, axis = 1, keepdims=True)
    y = ex / sum_ex
    return y







