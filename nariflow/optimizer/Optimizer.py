class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self, del_grad = True):
        # params 중 none이 아닌 실제 파라미터들만 엄선하여 모아둔다.
        params = [p for p in self.target.params() if p.grad is not None]
        # 전처리(옵션)
        for f in self.hooks:
            f(params)
        # 매개 변수를 업데이트 한다.
        for param in params:
            self.update_one(param)
        # 업데이트가 끝난 모든 그래디언트들은 삭제한다.
        if del_grad:
            for param in params:
                param.grad = None

    # 껍데기 업데이트 함수
    def update_one(self, params):
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)