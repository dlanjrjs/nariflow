from .Optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, eta = 1e-8):
        super().__init__()
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = dict()
        self.v = dict()
        self.t = 0
        self.eta = eta

    def update(self, del_grad = True):
        # params 중 none이 아닌 실제 파라미터들만 엄선하여 모아둔다.
        params = [p for p in self.target.params() if p.grad is not None]
        # 전처리(옵션)
        for f in self.hooks:
            f(params)
        # 매개 변수를 업데이트 한다.
        for i, param in enumerate(params):
            self.update_one(param, i)
        self.t += 1
        # 업데이트가 끝난 모든 그래디언트들은 삭제한다.
        if del_grad:
            for param in params:
                param.grad = None

    def update_one(self, param,i):
        if self.t == 0:
            self.m[i] = np.zeros_like(param.grad.data)
            self.v[i] = np.zeros_like(param.grad.data)

        self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * param.grad.data
        self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (param.grad.data ** 2)

        param.data -= self.m[i] * (self.lr / (self.v[i] + self.eta) ** (1/2))
