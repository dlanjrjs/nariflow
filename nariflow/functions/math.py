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

class Sign(Function):
    def forward(self, x):
        y = np.sign(x)
        return y
    
    def backward(self, gy):
        return 0
    
def sign(x):
    return Sign()(x)


class Max(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        y = np.max(x, axis=self.axis)
        return y

    def backward(self, gy):
        x = self.input_list[0].data
        if self.axis is not None:
            argmax = np.argmax(x, axis=self.axis).reshape(-1)
            shapes = x.shape
            gx = np.zeros(shapes)
            gy = gy.data.reshape(-1)

            shape_list = [list(range(0, i)) for i in shapes]
            shape_list.pop(self.axis)
            index_comb = np.array(np.meshgrid(*shape_list)).T.reshape(-1, len(shape_list))
            result_list = []
            for i, j in zip(index_comb, argmax):
                i = i.tolist()
                i.insert(self.axis, j)
                result_list.append(i)
            result_list = [[str(j) for j in i] for i in result_list]
            result_list = [','.join(i) for i in result_list]
            for i,j in enumerate(result_list):
                exec(f'gx[{j}] = gy[{i}]')

        if self.axis is None:
            gx = np.zeros(len(x.reshape(-1)))
            gx[np.argmax(x)] = int(gy.data)
            gx = gx.reshape(x.shape)
        gx = Variable(gx)
        return gx

def flowmax(x, axis = None):
    return Max(axis)(x)

class Min(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        y = np.min(x, axis=self.axis)
        return y

    def backward(self, gy):
        x = self.input_list[0].data
        if self.axis is not None:
            argmin = np.argmin(x, axis=self.axis).reshape(-1)
            shapes = x.shape
            gx = np.zeros(shapes)
            gy = gy.data.reshape(-1)

            shape_list = [list(range(0, i)) for i in shapes]
            shape_list.pop(self.axis)
            index_comb = np.array(np.meshgrid(*shape_list)).T.reshape(-1, len(shape_list))
            result_list = []
            for i, j in zip(index_comb, argmin):
                i = i.tolist()
                i.insert(self.axis, j)
                result_list.append(i)
            result_list = [[str(j) for j in i] for i in result_list]
            result_list = [','.join(i) for i in result_list]
            for i,j in enumerate(result_list):
                exec(f'gx[{j}] = gy[{i}]')

        if self.axis is None:
            gx = np.zeros(len(x.reshape(-1)))
            gx[np.argmin(x)] = int(gy.data)
            gx = gx.reshape(x.shape)
        gx = Variable(gx)
        return gx

def flowmin(x, axis = None):
    return Min(axis)(x)

class Absolute(Function):
    def forward(self, x):
        y = np.abs(x)
        return y
    def backward(self, gy):
        x = self.input_list[0]
        if x.data < 0:
            return -1 * gy
        elif x.data > 0:
            return 1 * gy
        else :
            raise Exception("Absolute value Couldn't define derivative at zero")

def flowabs(x):
    return Absolute()(x)