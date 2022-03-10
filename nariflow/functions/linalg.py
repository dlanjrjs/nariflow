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
from . import math

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


class TriangleLow(Function):
    def __init__(self, k):
        if k is not None:
            self.k = k
        else:
            self.k = 0

    def forward(self, x):
        y = np.tril(x, k=self.k)
        return y

    def backward(self, gy):
        return gy

def trilower(x):
    return TriangleLow()(x)

class CopyLowtoUpper(Function):
    def forward(self, x):
        y = np.tril(x)
        y_lower = np.tril(x, k=-1)
        y = y + np.transpose(y_lower)
        return y

    def backward(self, gy):
        return gy

def copyltu(x):
    return CopyLowtoUpper()(x)

class Symmetric(Function):

    def forward(self, x):
        y = (1 / 2) * (x + np.transpose(x))
        return y

    def backward(self, gy):
        return gy

def symmetric(x):
    return Symmetric()(x)


class MatrixInv(Function):
    def forward(self, x):
        self.cholesky = np.linalg.cholesky(x)
        y = np.linalg.inv(self.cholesky)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        Lt = transpose(matinv(self.cholesky))
        gx = -2 * trilower(matmul(x, matmul(symmetric(gy), Lt)))
        return gx

def matinv(x):
    return MatrixInv()(x)


class CholeskyDecomp(Function):
    def forward(self, x):
        y = np.linalg.cholesky(x)
        self.cholesky = y
        return y

    def backward(self, gy):
        cholesky = Variable(self.cholesky)
        cholesky_transpose = transpose(cholesky)
        cholesky_inverse = matinv(cholesky)
        cholesky_inverse_transpose = transpose(cholesky_inverse)
        gx = (1 / 2) * matmul(cholesky_inverse_transpose,
                                  matmul(copyltu(matmul(cholesky_transpose, gy)),
                                         cholesky_inverse))
        return gx

def cholesky(x):
    return CholeskyDecomp()(x)


class LQDecomp(Function):
    def forward(self, x):
        # There's no LQDecomp in numpy
        Q, L = np.linalg.LQDecomp(x)
        self.Q = Q
        self.L = L
        return Q, L

    def backward(self, gy):
        gx_Q = gy[0]
        gx_L = gy[1]
        L = Variable(self.L)
        Q = Variable(self.Q)
        L_transpose = transpose(L)
        L_transpose_inverse = matinv(L_transpose)
        Q_transpose = transpose(Q)
        M = matmul(L_transpose, gx_L) - matmul(gx_Q, Q_transpose)
        gx = matmul(L_transpose_inverse, gx_Q + matmul(copyltu(M), Q))
        return gx

def _lqdecomp(x):
    return LQDecomp()(x)


class SVDecomp(Function):
    def __init__(self, eta):
        self.eta = eta

    def hfunc(self, x, eta):
        eta_mat = np.broadcast_to(np.array(eta), shape=x.shape())
        return math.flowmax([math.flowabs(x).data, eta_mat], axis=0) * (
                math.sign(x) + eta)

    def forward(self, x):
        y = np.linalg.svd(x, full_matrices = False)
        self.U = y[0]
        self.S = y[1]
        self.V = y[2]
        return Variable(y)

    def backward(self, gy):
        gx_U = gy[2]
        gx_S = diagonal(gy[1])
        gx_V = gy[0]
        U = Variable(self.U)
        S = diagonal(Variable(self.S))
        S_reduced = diagonal(S)
        V = Variable(self.V)

        G_1 = matmul(gx_U, transpose(U)) + (
                    matmul(matinv(S),
                           matmul(gx_V,
                                  matmul(transpose(V), S))))

        indicate = Variable(np.ones(G_1.shape())) - Variable(np.identity(G_1.shape()[0]))
        h = broadcast_to(S_reduced, shape=(S_reduced.shape()[-1],
                                           S_reduced.shape()[-1]))
        E = indicate / (self.hfunc(h - reshape(S_reduced, [-1, 1]), eta = self.eta) * (
                        self.hfunc(h + reshape(S_reduced, [-1, 1]), eta = self.eta)))

        G_2_b = matmul(matinv(S), matmul(gx_V, transpose(V)))
        identity = diagonal(diagonal(G_2_b))
        G_2 = gx_S + 2 * matmul(symmetric(G_1 * E), S) - identity

        gx_1 = matmul(matinv(S), gx_V)
        gx = matmul(transpose(U), matmul(G_2, V) + gx_1)
        return gx

def svd(x, eta = 1e-8):
    return SVDecomp(eta)(x)

class Norm(Function):
    def __init__(self, ord):
        self.ord = ord

    def forward(self, x):
        y = np.linalg.norm(x, ord=self.ord)
        self.y = y
        return y

    def backward(self, gy):
        x = self.input_list[0]
        gx = ((math.flowabs(x) / self.y) ** (self.ord - 1)) * math.sign(x) * gy
        return gx

def norm(x, ord = 2):
    return Norm(ord)(x)