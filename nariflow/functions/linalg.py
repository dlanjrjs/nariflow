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

class Concatenate(Function):
    def __init__(self, axis=None):
        if axis is None:
            self.axis = 0
        else:
            self.axis = axis

    def forward(self, x):
        x = [i.data for i in x]
        y = np.concatenate(x, axis=self.axis)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        shapes = [len(i) for i in x.data]
        concats = list()
        for i in shapes:
            temp = gy[:i]
            gy = gy[i:]
            concats.append(temp)
        return tuple(concats)

class Stack(Function):
    def __init__(self, axis=None):
        if axis is None:
            self.axis = 0
        else:
            self.axis = axis

    def forward(self, x):
        x = [i.data for i in x]
        y = np.stack(x, axis=self.axis)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        shapes = gy.shape()
        concats = list()
        for axis in range(len(x)):
            ind = ''.join(([':,' if i != self.axis else f'{axis},' for i, j in enumerate(shapes)]))
            temp = eval(f'gy[{ind[:-1]}]')
            concats.append(temp)
        return tuple(concats)


class DiagonalMat(Function):
    def forward(self, x):
        self.x_shape = x.shape
        y = np.diag(x)
        return y

    def backward(self, gy):
        gx = diagmat(gy)
        return gx

def flowconcat(x, axis = None):
    return Concatenate(axis)(x)

def flowstack(x, axis = None):
    return Stack(axis)(x)

def diagonal(x):
    return DiagonalMat()(x)