import numpy as np
from ..layer.Linear import Linear
from ..functions.linalg import broadcast_to, reshape, transpose
from ..functions import activation
from ..core.elementary_function import flowsum
from .Layer import Layer
import os
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../conf.txt")
with open(path, mode = 'r') as f:
    conf = f.readline()
    if conf == 'node':
        from ..core.core import Variable, Function, CalcGradient
    if conf == 'tape':
        from ..core.core_tape import Variable, Function, GradientTape
from ..core.elementary_function import Parameter

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.
        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.
        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = activation.tanh(self.x2h(x))
        else:
            h_new = activation.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.h2u = Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = activation.sigmoid(self.x2f(x))
            i = activation.sigmoid(self.x2i(x))
            o = activation.sigmoid(self.x2o(x))
            u = activation.tanh(self.x2u(x))
        else:
            f = activation.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = activation.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = activation.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = activation.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * activation.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new
