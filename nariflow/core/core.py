import numpy as np
import weakref

class Variable():
    """ # Variable의 역할
        ## 현재 변수의 현단계 노드 함수(바로 앞에서 수행한 연산 함수)를 저장한다
        ## 현재 변수의 뒤에서 전달해온 gradient값을 저장한다(gy_list)
        ## 현재 변수의 gradient값을 연산한다(func.backward() gy와 backward 결과값간 곱을 수행한다.) -> gx_list
        ## 현재 변수의 gradient값을 저장한다(gx_list -> 각 Variable.grad 에 저장)
        ## 현재 변수의 세대를 저장한다.(self.generation)"""


    def __init__(self, data):
        self.data = data
        self.grad = None
        self.node_function = None
        self.generation = 0

    def set_node_function(self, func):
        # Variable을 생성한 노드 함수를 기억한다.
        # 자동 미분시 그래프 관계를 구축하기 위해 사용한다.
        self.node_function = func
        # generation은 하나의 노드에 두개 이상의 엣지가 연결되어 있을 때(즉, a - b / a - c)
        # b와 c의 역전파(grad)를 가장 먼저 처리하고 a로 역전파를 시행하기 위해 필요한 값이다
        # 즉, b와 c는 a보다 큰 generation값을 동일하게 보유하기 때문에
        # generation 값 기준으로 역전파를 처리하게 되면 반드시 a보다는 먼저 처리하는 것이 보장된다.
        self.generation = func.generation + 1

    def resetgrad(self):
        self.grad = None

    def shape(self):
        return self.data.shape

    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

class Function():
    """
    # Function 껍데기 함수의 역할
    ## 현재 연산 함수 기준 입력측 변수값을 저장한다(x_list)
    ## 현재 연산 함수 기준 출력측 변수값을 저장한다(y_list)
    ## Variable에 현재 함수를 마킹한다(output.set_node_function(self))
    ## 다음 세대 Variable에 알려줄 세대를 기록한다.
    """

    def __call__(self, *inputs):
        # 단순 스칼라값일때 array 형태로 변환해주는 내부함수
        def as_array(x):
            if np.isscalar(x):
                return np.array(x)
            return x

        # Variable 형태가 아닐 때 Variable 형태로 변환해재는 내부함수
        def as_variable(x):
            if isinstance(x, Variable):
                return x
            return Variable(x)

        # Variable을 가져온다.
        inputs = [as_variable(x) for x in inputs]
        x_list = [as_array(i.data) for i in inputs]
        # forward 실시, 이 때 x_list는 이미 i.data로 내부 scalar값들이 노출된 상태다.
        y_list = self.forward(*x_list)
        # 출력 결과물이 tuple이 아니면 tuple로 변환해준다.
        if not isinstance(y_list, tuple):
            y_list = (y_list,)
        # 정전파 함수의 연산 결과를 Variable로 변환해준다.
        output_list = [Variable(as_array(y)) for y in y_list]

        # 현재 함수의 generation을 정의한다.
        self.generation = np.max([i.generation for i in inputs])
        # 현재 함수를 출력측 Variable의 node_function에 마킹한다.
        for output in output_list:
            output.set_node_function(self)
        self.input_list = inputs
        self.output_list = [weakref.ref(x) for x in output_list]
        if len(output_list) > 1:
            return output_list
        else:
            return output_list[0]

    # 정전파 껍데기 함수
    def forward(self, x_list):
        raise NotImplementedError()

    # 역전파 껍데기 함수
    def backward(self, gy_list):
        raise NotImplementedError()

class CalcGradient():
    def __call__(self, inputs):
        def as_array(x):
            if np.isscalar(x):
                return np.array(x)
            return x

        def as_variable(x):
            if isinstance(x, Variable):
                return x
            return Variable(x)

        def add_func(func):
            if func not in seen_set:
                funcs.append(func)
                seen_set.add(func)
                funcs.sort(key=lambda x: x.generation)
        x_list = [as_array(i.data) for i in inputs]
        for i,j in zip(inputs, x_list):
            i.grad = Variable(np.ones_like(j))

        seen_set = set()
        funcs = []

        # add_func는 variable에 저장되어있는 세대 정보(0,1,2,....)를 기준으로
        # Function을 정렬한다.
        # 이는 노드가 분할되어 하나의 노드에 엣지가 두개 연결될 시(a - b & a - c)
        # b와 c를 먼저!!!! 처리하고 그 이후에 a로 역전파 되도록 보장한다.

        # 스택 탐색 알고리즘을 이용하여 가장 뒷 단계 Variable에서 정의된 노드 함수를 가져온다.
        # 즉, 역전파를 시작하기 위한 초기값을 준비한다.
        for i in inputs:
            add_func(i.node_function)

        while funcs:
            func = funcs.pop()
            # 현재단계의 뒤에서 전달해온 gradient값을 리스트로 저장한다.
            gy_list = [output().grad for output in func.output_list]
            # 뒷단계 gradient를 노드 함수(즉, 현 연산 함수)에 투입하여 현재단계 grad를 구한다.
            gx_list = func.backward(*gy_list)
            if not isinstance(gx_list, tuple):
                gx_list = (gx_list,)

            # 현 단계의 Variable을 불러와 Variable.grad에 Gradient 연산 결과물을 저장한다.
            for x, gx in zip(func.input_list, gx_list):
                # 동일 변수 사용시 gx가 단순히 덧씌워지는 문제를 해결한다.
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # 현 단계에 연결되어 있는 노드 함수(즉, 앞 단계 연산 함수)를 다음 node_function으로
                # 저장한다.
                if x.node_function is not None:
                    add_func(x.node_function)

def calc_gradient(inputs):
    return CalcGradient()(inputs)