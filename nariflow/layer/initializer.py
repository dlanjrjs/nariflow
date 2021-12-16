import numpy as np
import weakref

def xavier_normal(I, O, dtype):
    var = np.sqrt(2 / (I + O))
    w = np.random.normal(0, var, size = (I, O)).astype(dtype) * np.sqrt(1 / I)
    return w

def xavier_uniform(I, O, dtype):
    lower = - np.sqrt(6 / (I + O))
    upper = np.sqrt(6 / (I + O))
    w = np.random.uniform(lower, upper, size = (I, O)).astype(dtype) * np.sqrt(1 / I)
    return w

def he_normal(I, O, dtype):
    var = np.sqrt(2 / I)
    w = np.random.normal(0, var, size = (I, O)).astype(dtype) * np.sqrt(1 / I)
    return w

def he_uniform(I, O, dtype):
    lower = - np.sqrt(6 / I)
    upper = np.sqrt(6 / I)
    w = np.random.uniform(lower, upper, size = (I,O)).astype(dtype) * np.sqrt(1 / I)
    return w

def initializer_loader():
    initializer = {'xavier_normal': xavier_normal,
                   'xavier_uniform' : xavier_uniform,
                   'he_normal' : he_normal,
                   'he_uniform' : he_uniform}
    return initializer