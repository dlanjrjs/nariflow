import os
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../conf.txt")
with open(path, mode = 'r') as f:
    conf = f.readline()
    if conf == 'node':
        from .core import Variable, Function, CalcGradient
    if conf == 'tape':
        from .core_tape import Variable, Function, GradientTape
from .thirdparty import sum_to
import numpy as np


class Reshape(Function):
    def __init__(self, shape):
        # 변환을 원하는 모양을 지정한다.
        self.shape = shape

    def forward(self, x):
        # 원본 모양을 저장해준다.
        self.x_shape = x.shape
        # 모양을 reshpae로 바꾼다.
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        # 역전파시에는 원본 모양을 복원한다.
        gx = reshape(gy, self.x_shape)
        return gx


def reshape(x, shape):
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, shape=None):
        self.shape = shape

    def forward(self, x):
        # 원본 텐서의 모양을 저장해둔다.
        # transpose를 실시한다.
        y = np.transpose(x, self.shape)
        return y

    def backward(self, gy):
        if self.shape is None:
            gx = transpose(gy)
        # Variable 변수를 받으므로, np.transpose가 아닌 GoteoFlow의 transpose를 받는다.
        # 저장되어있던 원본 모양을 복원한다.
        else:
            axes_len = len(self.shape)
            inv_axes = tuple(np.argsort([ax % axes_len for ax in self.shape]))
            gx = transpose(gy, shape=inv_axes)
        return gx

def transpose(x, shape=None):
    return Transpose(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        # 모양을 바꾸면서 덧연산을 실시할 목표 모양을 지정한다.
        self.shape = shape

    def forward(self, x):
        # 원본 모양을 기억한다.
        # sum_to 함수로 모양을 바꾸며 합을 실시한다.
        if isinstance(self.shape, tuple):
            y = sum_to(x, self.shape)
        else :
            y = sum_to(x, self.shape())
        return y

    def backward(self, gy):
        self.x_shape = self.input_list[0].data.shape
        # 역전파시엔 sum_to로 인해 바뀌었던 모양을 원본 모양으로 복원한다.
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sumto(x, shape):
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        x = self.input_list[0]
        gx = sumto(x, self.x_shape)
        return gx


def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    # 정전파 : 슬라이싱을 수행한다.
    def forward(self, x):
        y = x[self.slices]
        return y

    # 역전파 : 입력 크기와 슬라이싱 정보를 역함수인 GetItemGrad에 전달한다.
    def backward(self, gy):
        x = self.input_list[0]
        if not isinstance(x.data, (tuple, list)):
            f = GetItemGrad(self.slices, x.data.shape)
            return f(gy)
        else :
            return gy

def get_item(x, slices):
    return GetItem(slices)(x)

#GetItemGrad는 GetItem의 역함수다.
class GetItemGrad(Function):
    def __init__(self, slices, shape):
        self.slices = slices
        self.shape = shape

    #정전파(슬라이싱의 역전파) : 입력 크기만큼의 0행렬을 생성한 후, 슬라이싱 위치에 gy를 채워 반환한다.
    # 슬라이싱에서 잘린 성분은 0 그대로 남는다.
    def forward(self, gy):
        gx = np.zeros(self.shape)
        np.add.at(gx, self.slices, gy)
        return gx

    # 역전파(슬라이싱) : 슬라이싱을 수행한다.
    def backward(self, ggx):
        return get_item(ggx, self.slices)