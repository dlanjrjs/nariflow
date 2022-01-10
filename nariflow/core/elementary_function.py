import os
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../conf.txt")
with open(path, mode = 'r') as f:
    conf = f.readline()
    if conf == 'node':
        from .core import Variable, Function, CalcGradient
    if conf == 'tape':
        from .core_tape import Variable, Function, GradientTape

from . import shape_function as shape_f
from .thirdparty import reshape_sum_backward
import numpy as np

# 덧셈
class Add(Function):
    # 정전파 : 두 변수를 더한다.
    def forward(self, x_0, x_1):
        self.x_0_shape = x_0.shape
        self.x_1_shape = x_1.shape
        y = x_0 + x_1
        return y

    # 역전파 : 뒷 단계에서 들어온 그레디언트를 양쪽으로 균등하게 흘려보낸다.
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x_0_shape != self.x_1_shape:
            gx0 = shape_f.sumto(gx0, self.x_0_shape)
            gx1 = shape_f.sumto(gx1, self.x_1_shape)
        return gx0, gx1


# 곱셈
class Mul(Function):
    # 정전파 : 두 변수를 곱한다.
    def forward(self, x_0, x_1):
        y = x_0 * x_1
        return y

    # 역전파 : 방향을 스위치해서 뒷 단계의 그레디언트와 입력 변수를 곱해 흘려보낸다.
    def backward(self, gy):
        x_0 = self.input_list[0]
        x_1 = self.input_list[1]
        self.x_0_shape = x_0.data.shape
        self.x_1_shape = x_1.data.shape
        gx0, gx1 = gy, gy
        x0 = x_1 * gx1
        x1 = x_0 * gx0
        if self.x_0_shape != self.x_1_shape:
            x0 = shape_f.sumto(x0, self.x_0_shape)
            x1 = shape_f.sumto(x1, self.x_1_shape)
        return x0, x1

    # 음수 변환


class Neg(Function):
    # 정전파 : 음수로 바꾼다.
    def forward(self, x):
        return -x

    # 역전파 : 음수로 바꿔 흘려보낸다.
    def backward(self, gy):
        return -gy


# 뺄셈
class Sub(Function):
    # 정전파 : 두 변수를 뺀다.
    def forward(self, x_0, x_1):
        self.x_0_shape = x_0.shape
        self.x_1_shape = x_1.shape
        y = x_0 - x_1
        return y

    # 역전파 : 앞 변수는 그레디언트를, 뒤 변수는 그레디언트 음수를 흘려보낸다.
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x_0_shape != self.x_1_shape:
            gx0 = shape_f.sumto(gx0, self.x_0_shape)
            gx1 = shape_f.sumto(gx1, self.x_1_shape)
        return gx0, -gx1


# 나눗셈
class Div(Function):
    # 정전파 : 변수간 나눗셈을 구한다.
    def forward(self, x_0, x_1):
        self.x_0_shape = x_0.shape
        self.x_1_shape = x_1.shape
        y = x_0 / x_1
        return y

    # 역전파 : 앞 변수의 경우 1 / a를, 뒤 변수의 경우 (- a / b **2)를 그레디언트와 곱해 흘려보낸다.
    def backward(self, gy):
        x_0, x_1 = self.input_list
        gx0, gx1 = gy, gy
        if self.x_0_shape != self.x_1_shape:
            gx0 = shape_f.sumto(gx0, self.x_0_shape)
            gx1 = shape_f.sumto(gx1, self.x_1_shape)
        gx_0 = (1 / x_1) * gx0
        gx_1 = (- x_0 / (x_1) ** 2) * gx1
        return gx_0, gx_1

    # 거듭제곱


class Pow(Function):
    # Function 클래스에 거듭제곱 수를 init으로 정의한다.
    def __init__(self, power):
        self.power = power

    # 정전파 : 변수에 거듭제곱을 한다.
    def forward(self, x):
        y = x ** self.power
        return y

    # 역전파 : power * x ^ (power - 1) 에 그레디언트를 곱해 흘려보낸다.
    def backward(self, gy):
        x = self.input_list[0]
        gx = self.power * x ** (self.power - 1) * gy
        return gx


def add(x_0, x_1):
    return Add()(x_0, x_1)


def mul(x_0, x_1):
    return Mul()(x_0, x_1)


def neg(x):
    return Neg()(x)


def sub(x_0, x_1):
    return Sub()(x_0, x_1)


def rsub(x_0, x_1):
    return Sub()(x_1, x_0)


def div(x_0, x_1):
    return Div()(x_0, x_1)


def rdiv(x_0, x_1):
    return Div()(x_1, x_0)


def power(x, power):
    return Pow(power)(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy, shape = reshape_sum_backward(gy,
                              x_shape=self.x_shape,
                              axis=self.axis,
                              keepdims=self.keepdims)
        gy = shape_f.reshape(gy, shape)
        gx = shape_f.broadcast_to(gy, self.x_shape)
        return gx

class MatMul(Function):
    def forward(self, x, w):
        y = x.dot(w)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        w = self.input_list[1]
        gw = matmul(shape_f.transpose(x), gy)
        gx = matmul(gy, shape_f.transpose(w))
        return gx, gw

def flowsum(x, axis = None, keepdims = False):
    return Sum(axis, keepdims)(x)

def matmul(x, w):
    return MatMul()(x, w)

class Parameter(Variable):
    pass


# 연산 기본 메소드들을 덮어씌워
# Variable과 연관된 연산은 기호(+, -, *, /)만 사용해도
# 우리가 정의한 연산을 수행하도록 대치한다.
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = power
    Variable.__getitem__ = shape_f.get_item

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        gx = exp(x) * gy
        return gx

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        gx = (1 / x) * gy
        return gx

def exp(x):
    return Exp()(x)

def log(x):
    return Log()(x)

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        gx = cos(x) * gy
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        return -sin(x) * gy

def cos(x):
    return Cos()(x)

class StopGradient(Function):
    def forward(self, x):
        return x

    def backward(self, gy):
        return 0

def stop_gradient(x):
    return StopGradient()(x)
