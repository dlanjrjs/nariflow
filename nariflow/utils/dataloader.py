import numpy as np

class DataSet():
    def __init__(self, data, label = None):
        self.data = data
        if label is not None:
            self.label = label
        else :
            self.label = None
        self.i = 0

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.data[index], None
        else :
            return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.batch_size is None:
            raise Excpetion('Please set up batch_size with batch_setup method')
        train = self.data[(self.batch_size * self.i) : (self.batch_size * (self.i + 1))]
        if len(train.data) == 0:
            raise StopIteration
        if self.label is not None:
            label = self.label[(self.batch_size * self.i) : (self.batch_size * (self.i + 1))]
            self.i += 1
            return train, label
        self.i += 1
        return train

    def batch_setup(self, batch_size, reset = True):
        if reset:
            self.reset()
        self.batch_size = batch_size

    def reset(self):
        self.i = 0


