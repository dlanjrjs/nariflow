from ..core.elementary_function import Parameter
from .initializer import initializer_loader
import weakref

class Layer():
    def __init__(self):
        self._params = set()
        self.initializer = initializer_loader()

    def __setattr__(self, name, value):
        # value가 Parameter 인스턴스일 경우엔 해당 인스턴스(레이어)의 name을 저장한다.
        # layer에 신규 paramter가 등록되면, 해당 인스턴스명을 _params에 보관한다.
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        # 기본 구조는 Function 껍데기 함수와 동일하다.
        # Function과 마찬가지로 해당 Layer에 들어온 입력과 출력을 저장한다.
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]

        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]

    def forward(self,x):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            # self.__dict__ 인스턴스에는 사용자가 Layer.{} 형태로 등록한 실제 파라미터들이 저장된다.
            # self._params에서 {} 인스턴스명을 하나하나씩 꺼내와 실제 파라미터값을 반환한다.
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def resetgrads(self):
        for param in self.params():
            param.resetgrad()
