import numpy as np
import pandas as pd
try:
   import cupy as cp
except Exception as e:
   print(e)

from .. import Variable
from .. import optimizer
from .. import GradientTape
from .. import layer
from ..models import Model
from .. import functions as f
from ..core import elementary_function as ef
from ..core import shape_function as sf
from . import DataSet

import tensorflow as tf
from tensorflow.keras import datasets
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

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

class clf_Models(Model):
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
        y = f.activation.softmax(y)
        return y


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = layer.RNN(hidden_size)
        self.l2 = layer.Linear(out_size, initializer_func='he_uniform')

    def forward(self, x):
        y = self.l1(x)
        y = y[-2:-1]
        y = self.l2(y)
        return y

class CNN(Model):
    def __init__(self):
        super().__init__()
        self.l1 = layer.Conv2d(64, (3,3))
        self.l2 = layer.Conv2d(128, (2,2))
        self.l3 = layer.Conv2d(64, (3,3))
        self.predict = layer.Linear(10)
        self.flatten = layer.flatten

    def forward(self, x):
        y = self.l1(x)
        y = f.activation.relu(y)
        y = self.l2(x)
        y = f.activation.relu(y)
        y = self.l3(x)
        y = f.activation.relu(y)
        y = self.flatten(y)
        y = self.predict(y)

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

class LinalgFunction():
    def linalg_concat_test(self):
        x = Variable(np.array([[[3.0, 2.0, 1.0]], [[6., 5., 4.]]]))
        y = Variable(np.array([[[1.0, 2.0, 3.0]], [[4., 5., 6.]]]))
        k = Variable(np.array([[[1.0, 2.0, 5.0]], [[7., 8., 9.]]]))

        with GradientTape() as tape:
            d = x * y
            concats = f.linalg.flowconcat([k, d])
            result = concats * concats

        tape.CalcGradient()

        return concats.grad.data, k.grad.data, d.grad.data, x.grad.data, y.grad.data

    def linalg_stack_test(self):
        x = Variable(np.array([[[3.0, 2.0, 1.0]], [[6., 5., 4.]]]))
        y = Variable(np.array([[[1.0, 2.0, 3.0]], [[4., 5., 6.]]]))
        k = Variable(np.array([[[1.0, 2.0, 5.0]], [[7., 8., 9.]]]))

        with GradientTape() as tape:
            d = x * y
            concats = f.linalg.flowstack([k, d])
            result = concats * concats

        tape.CalcGradient()

        return concats.grad.data, k.grad.data, d.grad.data, x.grad.data, y.grad.data

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

    def linalg_concat_test(self):
        a = np.array([[[2., 4., 10.]],
                      [[14., 16., 18.]],
                      [[6., 8., 6.]],
                      [[48., 50., 48.]]])

        b = np.array([[[2., 4., 10.]],
                      [[14., 16., 18.]]])

        c = np.array([[[6., 8., 6.]],
                      [[48., 50., 48.]]])

        d = np.array([[[6., 16., 18.]],
                      [[192., 250., 288.]]])

        e = np.array([[[18., 16., 6.]],
                      [[288., 250., 192.]]])

        return a, b, c, d, e

    def linalg_stack_test(self):
        a = np.array([[[[2., 4., 10.]],
                       [[14., 16., 18.]]],
                      [[[6., 8., 6.]],
                       [[48., 50., 48.]]]])

        b = np.array([[[2., 4., 10.]],
                      [[14., 16., 18.]]])

        c = np.array([[[6., 8., 6.]],
                      [[48., 50., 48.]]])

        d = np.array([[[6., 16., 18.]],
                      [[192., 250., 288.]]])

        e = np.array([[[18., 16., 6.]],
                      [[288., 250., 192.]]])

        return a, b, c, d, e


class UnitTest():
    def __init__(self):
        self.dataset_path = tf.keras.utils.get_file(
            "auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
        self.dataset_clf_path = tf.keras.utils.get_file(
            'petfinder_mini.zip', 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip',
                        extract=True, cache_dir='.')
        self.jacobian_preset = JacobianFunction()
        self.function_preset = TestFunction()
        self.order_preset = OrderFunction()
        self.start_preset = GradientStartFunction()
        self.answer_preset = TestAnswer()
        self.linalg_preset = LinalgFunction()

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
    def data_preprocessing_clf(self):
        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                        'Acceleration', 'Model Year', 'Origin']
        raw_dataset = pd.read_csv('datasets/petfinder-mini/petfinder-mini.csv')
        # In the original dataset "4" indicates the pet was not adopted.
        raw_dataset['target'] = np.where(raw_dataset['AdoptionSpeed'] == 4, 0, 1)
        # Drop un-used columns.
        train = raw_dataset.drop(columns=['AdoptionSpeed', 'Description'])
        train_processing = pd.concat([pd.get_dummies(train.loc[:, train.apply(lambda x: x.dtype == 'object')]),
                                      train.loc[:, train.apply(lambda x: x.dtype != 'object')].apply(
                                          lambda x: (x - min(x)) / (max(x) - min(x)))],
                                     axis=1)

        X = train_processing.drop(['target'], axis=1).to_numpy()
        y = np.array(pd.get_dummies(train_processing.loc[:, 'target']))

        return X, y

    def data_preprocessing_rnn(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        y = np.sin(x) + noise
        y = y.astype(dtype)
        data = y[:-1][:, np.newaxis]
        label = y[1:][:, np.newaxis]

        return data, label

        return data, label

    def data_preprocessing_cnn(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 픽셀 값을 0~1 사이로 정규화합니다.
        train_images, test_images = train_images / 255.0, test_images / 255.0

        train_images = train_images.reshape(train_images.shape[0],
                                            train_images.shape[3],
                                            train_images.shape[1],
                                            train_images.shape[2])

        test_images = test_images.reshape(test_images.shape[0],
                                          test_images.shape[3],
                                          test_images.shape[1],
                                          test_images.shape[2])

        return train_images, train_labels, test_images, test_labels

    def answer_correction(self, function, answer, pred, tor):
        answer = [np.abs(i.reshape(j.shape)) for i,j in zip(answer, pred)]
        pred = [np.abs(x) for x in pred]
        loss = [np.abs(i - j) for i,j in zip(answer, pred)]
        loss = np.array([i.sum() for i in loss])
        if np.all(loss < tor):
            print(function, ' : ok')
        else :
            print(function, ' : Failed')
            print('answer :', answer)
            print('pred :', pred)
            print('error :', loss)

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
            print(f'preset_function_test is failed :', e)

    def high_order_test(self, orders=2):
        #try:
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
            answer = [np.array(i) for i in answer]
            self.answer_correction(f'high_order_test_{order}', answer, pred, 0.01)
            tape.resetgrads()
        #except Exception as e:
        #    print(f'high_order_test_{order} is failed :', e)

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
                self.answer_correction(f'jacobian_{function}', answer, pred, tor)
                tape.resetgrads()
        except Exception as e:
            print(f'jacobian_test is failed :', e)

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
            print(f'gradient_{function} is failed :', e)

    def linalg_test(self, tor=0.01):
        try:
            linalg_function = self.linalg_preset
            function_list = [i for i in linalg_function.__dir__() if not i.startswith('_')]
            for function in function_list:
                current_function = linalg_function.__getattribute__(function)
                preds = current_function()
                answers = self.answer_preset.__getattribute__(function)()
                self.answer_correction(function, answers, preds, tor)
        except Exception as e:
            print(f'linalg_{function} is failed :', e)

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

    def modeling_test_clf(self, tor=1e-7, end_iter=4):
        try:
            X, y = self.data_preprocessing_clf()

            lr = 0.001
            hidden_size = 200
            out_size = 2
            batch_size = 50
            loss_flow = []
            loss_iter = 0

            X = Variable(np.array(X))
            y = Variable(y)


            start_time = time.time()
            model = clf_Models(hidden_size, out_size)
            optimizers = optimizer.Adam(lr)
            optimizers.setup(model)
            for i in range(10000):
                dataset = DataSet(X, y)
                dataset.batch_setup(batch_size)
                for j in range(np.ceil((len(dataset) / batch_size)).astype('int')):
                    X_set, y_set = next(dataset)

                    with GradientTape() as tape:
                        y_pred = model(X_set)
                        loss = f.loss.categorical_crossentropy(y_set, y_pred)
                    tape.CalcGradient()
                    optimizers.update()

                loss_flow.append(loss.data)
                if (loss_iter > end_iter):
                    if (abs(loss_flow[loss_iter - 1]) - (
                            abs(loss_flow[loss_iter]))) < tor:
                        print('loss_value :', [float(i) for i in loss_flow])
                        print('model is under local minima, Attempt to retry..')
                        self.modeling_test_clf()
                        break
                    else:
                        print('loss_value :', [float(i) for i in loss_flow])
                        print('clf model training test : ok')
                        break
                loss_iter += 1
        except Exception as e:
            print(f'clf model_training test is failed : {e}')

    def modeling_test_rnn(self, tor=1e-7, end_iter=4):
        try:
            X, y = self.data_preprocessing_rnn()

            step = 10
            lr = 0.001
            loss_flow = []
            loss_iter = 0

            start_time = time.time()
            model = SimpleRNN(100, 1)
            optimizers = optimizer.SGD()
            optimizers.setup(model)
            for i in range(10000):
                for teps in range(len(X) - step - 1):
                    with GradientTape() as tape:
                        y_pred = model(X[teps: teps + step])
                        loss = f.loss.mean_squared_error(y_pred, y[(teps + 1) + step])

                    tape.CalcGradient()
                    optimizers.update()

                loss_flow.append(loss.data)
                if (loss_iter > end_iter):
                    if (abs(loss_flow[loss_iter - 1]) - (
                            abs(loss_flow[loss_iter]))) < tor:
                        print('loss_value :', [float(i) for i in loss_flow])
                        print('model is under local minima, Attempt to retry..')
                        self.modeling_test_rnn()
                        break
                    else:
                        print('loss_value :', [float(i) for i in loss_flow])
                        y_pred = list()
                        for teps in range(int(len(X) - step)):
                            y_pred.append(model(X[teps: teps + step]).data)
                        plt.plot(np.squeeze(np.squeeze(y_pred)))
                        plt.plot(np.squeeze(X))
                        print('RNN model training test : ok')
                        break
                loss_iter += 1
        except Exception as e:
            print(f'RNN model_training test is failed : {e}')

    def modeling_test_cnn(self, tor=1e-7, end_iter=2):
        try:
            X_train, y_train, X_test, y_test = self.data_preprocessing_cnn()

            lr = 0.001
            loss_flow = []
            loss_iter = 0
            batch_size = 100

            start_time = time.time()
            model = CNN()
            optimizers = optimizer.Adam(lr)
            optimizers.setup(model)
            for i in range(10000):
                for j in range(np.ceil((len(X_train) / batch_size)).astype('int')):
                    train = X_train[j * batch_size: (j + 1) * batch_size]
                    label = y_train[j * batch_size: (j + 1) * batch_size]

                    with GradientTape() as tape:
                        y_pred = model(train)
                        loss = f.loss.softmax_cross_entropy(y_pred, label)

                    tape.CalcGradient()
                    optimizers.update()
                print(i)
                loss_flow.append(loss.data)
                if (loss_iter > end_iter):
                    if (abs(loss_flow[loss_iter - 1]) - (
                            abs(loss_flow[loss_iter]))) < tor:
                        print('loss_value :', [float(i) for i in loss_flow])
                        print('model is under local minima, Attempt to retry..')
                        self.modeling_test_cnn()
                        break
                    else:
                        print('loss_value :', [float(i) for i in loss_flow])
                        test_pred = model(X_test)
                        print('test셋 정확도 : ', accuracy_score(np.argmax(test_pred.data, axis=-1), y_test))
                        print('혼동행렬')
                        print(confusion_matrix(np.argmax(test_pred.data, axis=-1), y_test))
                        print('CNN model training test : ok')
                        break
                loss_iter += 1
        except Exception as e:
            print(f'CNN model_training test is failed : {e}')

    def start_testing(self):
        self.high_order_test()
        self.matrix_test()
        self.jacobian_test()
        self.gradient_start_index_test()
        self.linalg_test()
        self.modeling_test()
        self.modeling_test_clf()
        self.modeling_test_rnn()
        self.modeling_test_cnn()