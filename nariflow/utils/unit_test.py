import numpy as np
import pandas as pd
import cupy as cp

from .. import Variable
from .. import optimizer
from .. import GradientTape
from .. import layer
from ..models import Model
from .. import functions as f
from ..core import elementary_function as ef
from ..core import shape_function as sf

import tensorflow as tf

import time

class Models(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = layer.Linear(hidden_size, initializer_func='he_uniform')
        self.l2 = layer.Linear(hidden_size, initializer_func='he_uniform')
        self.l3 = layer.Linear(out_size, initializer_func='he_uniform')

    def forward(self, x):
        y = self.l1(x)
        y = f.activation.relu(y)
        y = self.l2(y)
        y = f.activation.relu(y)
        y = self.l3(y)
        return y


class TestFunction():
    def matyas(self, x, y):
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return z

    def goldstein(self, x, y):
        z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
        return z


class OrderFunction():
    def high_order_function(self, x, y):
        z = x ** 4 + y ** 3 + x ** 2 + y ** (1 / 2)
        return z

    def matrix_order_function(self, x):
        y = 2 * x ** 3
        return y


class JacobianFunction():
    def matmul(self, x, y, k):
        with GradientTape() as tape:
            result = ef.matmul(x, sf.transpose(y))
            result_2 = ef.matmul(result, k)
        return tape

    def reduce_sum(self, x, y, k):
        with GradientTape() as tape:
            result = ef.matmul(x, sf.transpose(y))
            result_2 = ef.matmul(result, k)
            result_3 = ef.flowsum(result_2)
        return tape

    def div(self, x, y, k):
        with GradientTape() as tape:
            result = ef.matmul(x, sf.transpose(y))
            result_2 = ef.matmul(result, k)
            result_3 = result_2 / result
        return tape

    def sum(self, x, y, k):
        with GradientTape() as tape:
            result = ef.matmul(x, sf.transpose(y))
            result_2 = ef.matmul(result, k)
            result_3 = result_2 + result
        return tape

    def mul(self, x, y, k):
        with GradientTape() as tape:
            result = ef.matmul(x, sf.transpose(y))
            result_2 = ef.matmul(result, k)
            result_3 = result_2 * result
        return tape


class GradientStartFunction():
    def gradient_start_middle(self, x, y, k):
        x = Variable(np.array([[1., 2.], [4., 5.]]))
        v = Variable(np.array([[4., 5.], [6., 7.]]))
        k = Variable(np.array([[1., 3.], [4., 6.]]))

        with GradientTape() as tape:
            result = ef.matmul(x, sf.transpose(v))
            result_2 = ef.matmul(result, k)
            result_3 = result_2 / x

        tape.CalcGradient(target=result_2)
        return x.grad.data, v.grad.data, k.grad.data

    def gradient_stop_test(self, x, y, k):
        x = Variable(np.array([[1., 2.], [4., 5.]]))
        v = Variable(np.array([[4., 5.], [6., 7.]]))
        k = Variable(np.array([[1., 3.], [4., 6.]]))

        with GradientTape() as tape_1, GradientTape() as tape_2:
            result = ef.matmul(x, sf.transpose(v))
            result_2 = ef.matmul(ef.stop_gradient(result), k)
            result_2 = result_2 ** 2
            result_3 = result_2 * x
        tape_2.CalcGradient()
        return x.grad.data, k.grad.data

class TestAnswer():

    def matyas(self, x=None, y=None):
        if (x is None) | (y is None):
            return (0.040000000000000036, 0.040000000000000036)
        else:
            return

    def goldstein(self, x=None, y=None):
        if (x is None) | (y is None):
            return (-5376.0, 8064.0)
        else:
            return

    def high_order_function(self, order, x=None, y=None):
        if order == 0:
            if (x is None) | (y is None):
                return 36., 12.3535
            else:
                return

        if order == 1:
            if (x is None) | (y is None):
                return 50, 11.911
            else:
                return

        else:
            return

    def matrix_order_function(self, order, x=None, y=None):
        if order == 0:
            if (x is None) | (y is None):
                return np.array([[6., 24.], [96., 150.]])
            else:
                return

        if order == 1:
            if (x is None) | (y is None):
                return np.array([[12., 24.], [48., 60.]])
            else:
                return

        if order == 2:
            if (x is None) | (y is None):
                return np.array([[12., 12.], [12., 12.]])
            else:
                return

    def matmul(self, x=None, y=None, k=None):
        answer = []
        if x is None:
            x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
            y = tf.Variable(np.array([[4., 5.], [6., 7.]]))
            k = tf.Variable(np.array([[1., 3.], [4., 6.]]))
            for i in ['x', 'y', 'k']:
                with tf.GradientTape() as tape:
                    result = tf.matmul(x, tf.transpose(y))
                    result_2 = tf.matmul(result, k)

                answer.append(tape.jacobian(result_2, eval(i)).numpy())
            return tuple(answer)

    def reduce_sum(self, x=None, y=None, k=None):
        if x is None:
            x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
            y = tf.Variable(np.array([[4., 5.], [6., 7.]]))
            k = tf.Variable(np.array([[1., 3.], [4., 6.]]))
            answer = []
            for i in ['x', 'y', 'k']:
                with tf.GradientTape() as tape:
                    result = tf.matmul(x, tf.transpose(y))
                    result_2 = tf.matmul(result, k)
                    result_3 = tf.math.reduce_sum(result_2)

                answer.append(tape.jacobian(result_3, eval(i)).numpy())

            return tuple(answer)

    def div(self, x=None, y=None, k=None):
        if x is None:
            x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
            y = tf.Variable(np.array([[4., 5.], [6., 7.]]))
            k = tf.Variable(np.array([[1., 3.], [4., 6.]]))
            answer = []
            for i in ['x', 'y', 'k']:
                with tf.GradientTape() as tape:
                    result = tf.matmul(x, tf.transpose(y))
                    result_2 = tf.matmul(result, k)
                    result_3 = result_2 / result

                answer.append(tape.jacobian(result_3, eval(i)).numpy())

            return tuple(answer)

    def sum(self, x=None, y=None, k=None):
        if x is None:
            x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
            y = tf.Variable(np.array([[4., 5.], [6., 7.]]))
            k = tf.Variable(np.array([[1., 3.], [4., 6.]]))
            answer = []
            for i in ['x', 'y', 'k']:
                with tf.GradientTape() as tape:
                    result = tf.matmul(x, tf.transpose(y))
                    result_2 = tf.matmul(result, k)
                    result_3 = result_2 + result

                answer.append(tape.jacobian(result_3, eval(i)).numpy())

            return tuple(answer)

    def mul(self, x=None, y=None, k=None):
        if x is None:
            x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
            y = tf.Variable(np.array([[4., 5.], [6., 7.]]))
            k = tf.Variable(np.array([[1., 3.], [4., 6.]]))
            answer = []
            for i in ['x', 'y', 'k']:
                with tf.GradientTape() as tape:
                    result = tf.matmul(x, tf.transpose(y))
                    result_2 = tf.matmul(result, k)
                    result_3 = result_2 * result

                answer.append(tape.jacobian(result_3, eval(i)).numpy())

            return tuple(answer)

    def start_middle(self, x=None, y=None, k=None):
        x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
        v = tf.Variable(np.array([[4., 5.], [6., 7.]]))
        k = tf.Variable(np.array([[1., 3.], [4., 6.]]))

        with tf.GradientTape() as tape:
            result = tf.matmul(x, tf.transpose(v))
            result_2 = tf.matmul(result, k)
            result_3 = result_2 / x

        answer = tape.gradient(result_2, [x, v])

        return answer

    def gradient_start_middle(self, x=None, y=None, k=None):
        x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
        v = tf.Variable(np.array([[4., 5.], [6., 7.]]))
        k = tf.Variable(np.array([[1., 3.], [4., 6.]]))

        with tf.GradientTape() as tape:
            result = tf.matmul(x, tf.transpose(v))
            result_2 = tf.matmul(result, k)
            result_3 = result_2 / x

        answer = tape.gradient(result_2, [x, v, k])
        answer = [i.numpy() for i in answer]

        return answer

    def gradient_stop_test(self, x=None, y=None, k=None):
        x = tf.Variable(np.array([[1., 2.], [4., 5.]]))
        v = tf.Variable(np.array([[4., 5.], [6., 7.]]))
        k = tf.Variable(np.array([[1., 3.], [4., 6.]]))

        with tf.GradientTape() as tape:
            result = tf.matmul(x, tf.transpose(v))
            result_2 = tf.matmul(tf.stop_gradient(result), k)
            result_2 = result_2 ** 2
            result_3 = result_2 * x

        answer = tape.gradient(result_3, [x, k])
        answer = [i.numpy() for i in answer]

        return answer


class UnitTest():
    def __init__(self):
        self.dataset_path = tf.keras.utils.get_file(
            "auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
        self.jacobian_preset = JacobianFunction()
        self.function_preset = TestFunction()
        self.order_preset = OrderFunction()
        self.start_preset = GradientStartFunction()
        self.answer_preset = TestAnswer()

    def data_preprocessing(self):
        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                        'Acceleration', 'Model Year', 'Origin']
        raw_dataset = pd.read_csv(self.dataset_path, names=column_names,
                                  na_values="?", comment='\t',
                                  sep=" ", skipinitialspace=True)

        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        origin = dataset.pop('Origin')

        dataset['USA'] = (origin == 1) * 1.0
        dataset['Europe'] = (origin == 2) * 1.0
        dataset['Japan'] = (origin == 3) * 1.0

        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        train_stats = train_dataset.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()

        normed_train_data = norm(train_dataset)
        normed_test_data = norm(test_dataset)
        train_labels = train_dataset.pop('MPG')

        X = Variable(np.array(normed_train_data.drop('MPG', axis=1)))
        y = (np.array(train_labels) - np.mean(np.array(train_dataset))) / np.std(np.array(train_dataset))
        y = Variable(np.array(y).reshape([-1, 1]))

        return X, y

    def answer_correction(self, function, answer, pred, tor):
        if len(np.array(answer).shape) > 1:
            answer = [i.reshape(j.shape) for i, j in zip(answer, pred)]
        if np.all(np.abs(answer) - np.abs(pred) < tor):
            print(function, ' : ok')
        else:
            print(function, ' : Failed')
            print('answer :', np.abs(answer))
            print('pred :', np.abs(pred))
            print('error :', np.abs(answer) - np.abs(pred))

    def preset_test(self, tor=0.01, function=None, x=None, y=None):
        try:
            if x is None:
                x = Variable(np.array(1.0))
            if y is None:
                y = Variable(np.array(1.0))
            function_list = [i for i in self.function_preset.__dir__() if not i.startswith('_')]
            for function in function_list:
                current_function = self.function_preset.__getattribute__(function)
                with GradientTape() as tape:
                    z = current_function(x, y)
                tape.CalcGradient()
                pred = (x.grad.data, y.grad.data)
                answer = self.answer_preset.__getattribute__(function)()
                self.answer_correction(function, answer, pred, tor)
                tape.resetgrads()
        except Exception as e:
            print(f'matrix_order_test_{order} is failed :', e)

    def high_order_test(self, orders=2):
        try:
            x = Variable(np.array(2.0))
            y = Variable(np.array(2.0))
            order_function = self.order_preset.high_order_function
            tape_dict = dict()
            with GradientTape() as tape:
                f = order_function(x, y)
            tape_dict[0] = tape
            for order in range(orders):
                with GradientTape() as tape_1:
                    tape_dict[order].CalcGradient()
                tape_dict[order + 1] = tape_1
                pred = (x.grad.data, y.grad.data)
                tape_dict[order].resetgrads()
                answer = self.answer_preset.high_order_function(order)
                self.answer_correction(f'high_order_test_{order}', answer, pred, 0.01)
                tape.resetgrads()
        except Exception as e:
            print(f'matrix_order_test_{order} is failed :', e)

    def matrix_test(self, orders=3):
        try:
            X = Variable(np.array([[1., 2.], [4., 5.]]))
            matrix_function = self.order_preset.matrix_order_function
            tape_dict = dict()
            with GradientTape() as tape:
                result = matrix_function(X)
            tape_dict[0] = tape
            for order in range(orders):
                with GradientTape() as tape_1:
                    tape_dict[order].CalcGradient()
                tape_dict[order + 1] = tape_1
                pred = X.grad.data
                tape_dict[order].resetgrads()
                answer = self.answer_preset.matrix_order_function(order)
                self.answer_correction(f'matrix_order_test_{order}',
                                       answer,
                                       pred,
                                       0.01)
        except Exception as e:
            print(f'matrix_order_test_{order} is failed :', e)

    def jacobian_test(self, tor=0.01, x=None, v=None, k=None, target=None):
        try:
            if x is None:
                x = Variable(np.array([[1., 2.], [4., 5.]]))
            if v is None:
                v = Variable(np.array([[4., 5.], [6., 7.]]))
            if k is None:
                k = Variable(np.array([[1., 3.], [4., 6.]]))
            jacobian_function = self.jacobian_preset
            function_list = [i for i in jacobian_function.__dir__() if not i.startswith('_')]
            for function in function_list:
                current_function = jacobian_function.__getattribute__(function)
                tape = current_function(x, v, k)
                if target is not None:
                    pred = (tape.jacobian(target=target, var=x, var_return='numpy'),
                            tape.jacobian(target=target, var=v, var_return='numpy'),
                            tape.jacobian(target=target, var=k, var_return='numpy'))
                else:
                    pred = (tape.jacobian(var=x, var_return='numpy'),
                            tape.jacobian(var=v, var_return='numpy'),
                            tape.jacobian(var=k, var_return='numpy'))
                answer = self.answer_preset.__getattribute__(function)()
                self.answer_correction(function, answer, pred, tor)
                tape.resetgrads()
        except Exception as e:
            print(f'matrix_order_test_{order} is failed :', e)

    def gradient_start_index_test(self, tor=0.01, x=None, v=None, k=None):
        try:
            if x is None:
                x = Variable(np.array([[1., 2.], [4., 5.]]))
            if v is None:
                v = Variable(np.array([[4., 5.], [6., 7.]]))
            if k is None:
                k = Variable(np.array([[1., 3.], [4., 6.]]))
            start_index_function = self.start_preset
            function_list = [i for i in start_index_function.__dir__() if not i.startswith('_')]
            for function in function_list:
                current_function = start_index_function.__getattribute__(function)
                pred = current_function(x, v, k)
                answer = self.answer_preset.__getattribute__(function)()
                self.answer_correction(function, answer, pred, tor)
        except Exception as e:
            print(f'{function} is failed :', e)

    def modeling_test(self, tor=1e-7, end_iter=4):
        try:
            X, y = self.data_preprocessing()

            lr = 0.1
            loss_flow = []
            loss_iter = 0

            start_time = time.time()
            model = Models(100, 1)
            optimizers = optimizer.Adam()
            optimizers.setup(model)
            for i in range(10000):
                with GradientTape() as tape:
                    y_pred = model(X)

                    loss = f.loss.mean_squared_error(y, y_pred)

                tape.CalcGradient()

                optimizers.update()

                if i % 1000 == 0:
                    loss_flow.append(loss.data)
                    if (loss_iter > end_iter):
                        if (abs(loss_flow[loss_iter - 1]) - (
                                abs(loss_flow[loss_iter]))) < tor:
                            print('loss_value :', [float(i) for i in loss_flow])
                            print('model is under local minima, Attempt to retry..')
                            self.modeling_test()
                            break
                        else:
                            print('loss_value :', [float(i) for i in loss_flow])
                            print('model training test : ok')
                            break
                    loss_iter += 1
        except Exception as e:
            print(f'model_training test is failed : {e}')

    def start_testing(self):
        self.high_order_test()
        self.matrix_test()
        self.jacobian_test()
        self.gradient_start_index_test()
        self.modeling_test()