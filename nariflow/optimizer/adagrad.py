from .Optimizer import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__()
        self.lr = lr
        self.h = 0

    def update_one(self, param):
        self.h += param.grad.data * param.grad.data
        h = self.h ** (1/2)
        param.data -= self.lr * (1 / h) * param.grad.data