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
from ..core.shape_function import reshape, sumto, broadcast_to, transpose
from ..core.elementary_function import matmul

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

def flowconcat(x, axis = None):
    return Concatenate(axis)(x)

def flowstack(x, axis = None):
    return Stack(axis)(x)

class Outer(Function):

    def forward(self, x1, x2):
        y = []
        if len(x1.shape) > 2 | len(x2.shape) > 2:
            raise Exception('Variable must not exceed tensor dimension over 2')
        for i, j in zip(x1, x2):
            y.append(np.outer(i, j))
        return np.array(y)

    def backward(self, gy):
        x1, x2 = self.input_list
        main_axis_i = x1.shape()[-1]
        main_axis_j = x2.shape()[-1]
        stack_var_x1 = Variable(np.array([]).reshape(0, main_axis_i))
        stack_var_x2 = Variable(np.array([]).reshape(0, main_axis_j))

        for i, j in zip(x1, x2):
            i = reshape(i, [-1, 1])
            j = reshape(j, [1, -1])

            i = i * main_axis_j
            j = j * main_axis_i

            gy_i = sumto(gy, (main_axis_i, 1))
            gy_j = sumto(gy, (1, main_axis_j))

            i = i * gy_i
            j = j * gy_j
            i = reshape(i, [1, -1])
            j = reshape(j, [1, -1])

            stack_var_x1 = flowconcat([stack_var_x1, i])
            stack_var_x2 = flowconcat([stack_var_x2, j])

        return stack_var_x1, stack_var_x2

def outer(x_1, x_2):
    return Outer()(x_1, x_2)

class DiagonalMat(Function):
    def forward(self, x):
        self.x_shape = x.shape
        y = np.diag(x)
        return y

    def backward(self, gy):
        gx = diagmat(gy)
        return gx


def diagonal(x):
    return DiagonalMat()(x)

class EigenVec(Function):

    def forward(self, x):
        self.eigval, self.eigvec = np.linalg.eig(x)
        return self.eigvec

    def backward(self, gy):
        x = self.input_list[0]
        second_term = diagonal(self.eigval) - x
        for num, vec in enumerate(self.eigvec):
            vec = broadcast_to(vec, second_term.shape())
            geigvec = outer(vec, second_term)
            if num == 0:
                stack_var = geigvec
            else :
                stack_var = flowconcat([stack_var, geigvec])
        gy = broadcast_to(gy, stack_var.shape())
        return stack_var * gy


class EigenVal(Function):
    def forward(self, x):
        eigval, eigvec = np.linalg.eig(x)
        self.eigval, self.eigvec = eigval, eigvec
        return eigval

    def backward(self, gy):
        eigvec = Variable(self.eigvec)
        geigval = outer(eigvec, eigvec)
        gy = broadcast_to(gy, eigvec.shape())
        return geigval * gy

def eigvec(x):
    return EigenVec()(x)

def eigval(x):
    return EigenVal()(x)