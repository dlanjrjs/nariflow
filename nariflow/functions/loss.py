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
from ..functions.activation import softmax
from ..thirdparty.functions import logsumexp


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
    y = y / pred.data.shape[0]
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.input_list
        N, CLS_NUM = x.shape()

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        #xp = cuda.get_array_module(t.data)
        xp = np
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype())[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)