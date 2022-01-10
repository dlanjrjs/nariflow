import numpy as np
import weakref


class Variable():
    def __init__(self, data):
        self.data = data
        self.generation = 0
        self.grad = None

    def set_generation(self, generation):
        self.generation = generation + 1

    def resetgrad(self):
        self.grad = None

    def shape(self):
        return self.data.shape

    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

class Function():
    def __call__(self, *inputs):
        def as_array(x):
            if np.isscalar(x):
                return np.array(x)
            return x

        def as_variable(x):
            if isinstance(x, Variable):
                return x
            return Variable(x)
        inputs = [as_variable(as_array(x)) for x in inputs]
        x_list = [i.data for i in inputs]
        y_list = self.forward(*x_list)
        if not isinstance(y_list, tuple):
            y_list = (y_list,)

        output_list = [Variable(as_array(y)) for y in y_list]
        generation = np.max([i.generation for i in inputs])
        for output in output_list:
            output.set_generation(generation)
        self.generation = generation
        if 'GRADIENT_TAPE' in globals():
            self.making_gradient_tape(output_list, inputs)

        if len(output_list) > 1:
            return output_list
        else:
            return output_list[0]

    def making_gradient_tape(self, output, inputs):
        for i in output:
            GRADIENT_TAPE[i] = (self, inputs, self.generation)

    def forward(self, x_list):
        raise NotImplementedError()

    def backward(self, gy_list):
        raise NotImplementedError()

class GradientTape():

    def __init__(self):
        global GRADIENT_TAPE
        GRADIENT_TAPE = dict()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        global GRADIENT_TAPE
        self.gradient_tape = GRADIENT_TAPE
        del GRADIENT_TAPE

    def CalcGradient(self, target = None, tapes = None):
        if tapes is None:
            tapes = self.gradient_tape
        tapes = dict(sorted(tapes.items(), key = lambda x : x[1][2]))

        if target is not None:
            target_ind = [i for i, j in enumerate([i == target for i in list(tapes)]) if j][0]
            tapes_dict = dict()
            [tapes_dict.update(i) for i in [{i[0]: i[1]} for i in tapes.items()][0:target_ind + 1]]
            tapes = dict(reversed(tapes_dict.items()))
        else :
            tapes = dict(reversed(tapes.items()))

        def as_array(x):
            if np.isscalar(x):
                return np.array(x)
            return x

        def as_variable(x):
            if isinstance(x, Variable):
                return x
            return Variable(x)

        for i, tape in enumerate(tapes.items()):
            outputs = tape[0]
            generation = tape[1][2]
            inputs = tape[1][1]
            func = tape[1][0]

            if i == 0:
                init_generation = generation.copy()

            if isinstance(outputs, Variable):
                outputs = [outputs]

            if generation == init_generation:
                for j in outputs:
                    j.grad = Variable(np.ones_like(j.data))
            gy_list = [output.grad for output in outputs]
            func.input_list = inputs
            gx_list = func.backward(*gy_list)
            if not isinstance(gx_list, tuple):
                gx_list = (gx_list,)

            for x, gx in zip(inputs, gx_list):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

    def resetgrads(self):
        for i in list(self.gradient_tape.keys()):
            i.grad = None

        for i in list(self.gradient_tape.values()):
            for j in i[1]:
                j.grad = None

    def jacobian(self, target = None, tapes = None, var = None, var_return = 'Variable'):
        if tapes is None:
            tapes = self.gradient_tape.copy()
        tapes = dict(sorted(tapes.items(), key=lambda x: x[1][2]))

        if target is not None:
            target_ind = [i for i, j in enumerate([i == target for i in list(tapes)]) if j][0]
            tapes_dict = dict()
            [tapes_dict.update(i) for i in [{i[0]: i[1]} for i in tapes.items()][0:target_ind + 1]]
            tapes = dict(reversed(tapes_dict.items()))
        else :
            tapes = dict(reversed(tapes.items()))
        def as_array(x):
            if np.isscalar(x):
                return np.array(x)
            if isinstance(x, type(np.array([]))):
                if len(x.shape) == 0:
                    return np.array([[x]])
            return x

        def as_variable(x):
            if isinstance(x, Variable):
                return x
            return Variable(x)
        if len(list(tapes.keys())[0].data.shape) >= 2:
            i_max = list(tapes.keys())[0].data.shape[0]
            j_max = list(tapes.keys())[0].data.shape[1]
        else :
            i_max = 1
            j_max = 1

        jacobian_dict = dict()
        for jacobian_iter_i in range(i_max):
            jacobian_dict_j = dict()
            for jacobian_iter_j in range(j_max):
                temp_dict = dict()
                for i, tape in enumerate(tapes.items()):
                    outputs = tape[0]
                    generation = tape[1][2]
                    inputs = tape[1][1]
                    func = tape[1][0]

                    if i == 0:
                        init_generation = generation.copy()

                    if isinstance(outputs, Variable):
                        outputs = [outputs]

                    if generation == init_generation:
                        for j in outputs:
                            j.data = as_array(j.data)
                            grad_matrix = np.zeros_like(j.data)
                            grad_matrix[jacobian_iter_i][jacobian_iter_j] = 1
                            j.grad = Variable(grad_matrix)

                    gy_list = [output.grad for output in outputs]
                    func.input_list = inputs
                    gx_list = func.backward(*gy_list)
                    if not isinstance(gx_list, tuple):
                        gx_list = (gx_list,)
                    for x, gx in zip(inputs, gx_list):
                        if x.grad is None:
                            x.grad = gx
                        else:
                            x.grad = x.grad + gx
                        temp_dict[x] = x.grad.data
                self.resetgrads()
                jacobian_dict_j[jacobian_iter_j] = temp_dict
            jacobian_dict[jacobian_iter_i] = jacobian_dict_j
        if var is None:
            return jacobian_dict

        selected_jacobian = list()
        for i in jacobian_dict:
            for j in jacobian_dict[i]:
                selected_jacobian.append(jacobian_dict[i][j][var])
        if var_return == 'numpy':
            return np.array(selected_jacobian)
        if var_return == 'list':
            return selected_jacobian
        if var_return == 'Variable':
            return [as_variable(x) for x in selected_jacobian]
        else :
            raise Exception('var_return only accpet "numpy", "list" or "Variable"')