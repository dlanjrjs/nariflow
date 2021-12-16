from .Layer import Layer
from ..core.elementary_function import Parameter, matmul
import numpy as np

def linear(x, W, b = None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y

class Linear(Layer):
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None, initializer_func = 'xavier_normal'):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.initializer_func = initializer_func

        # Parameter 인스턴스 생성
        self.W = Parameter(None)
        # 최초 생성시에 Input 사이즈가 지정되어 있을때만 init에서 초기화한다.
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else :
            self.b = Parameter(np.zeros(out_size, dtype = dtype))

    def _init_W(self):
        I = self.in_size
        O = self.out_size

        # 가중치가 폭발하지 않도록 가중치를 sqrt(1 / Input_size)로 정규화한다.
        W_data = self.initializer[self.initializer_func](I, O, dtype = self.dtype)
        self.W.data = W_data

    def forward(self, x):
        # 여전히 W가 None 상태라면(즉, Input Size가 init 단에서 초기화되지 않았다면)
        # 입력된 데이터를 기반으로 in_size를 정의하여 비로소 초기화한다.
        if self.W.data is None:
            self.in_size = x.data.shape[1]
            self._init_W()

        y = linear(x, self.W, self.b)
        return y