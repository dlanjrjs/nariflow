from .start_initialize import start_initializer
import os
#from .core import *
#from .thirdparty import *

my_path = os.path.abspath(os.path.dirname(__file__))
paths = os.path.join(my_path, "./conf.txt")
with open(paths, mode = 'r') as f:
    conf = f.readline()
    if conf == 'node':
        from .core.core import Variable, Function, calc_gradient
    if conf == 'tape':
        from .core.core_tape import Variable, Function, GradientTape

from .core.elementary_function import setup_variable, Parameter
from . import layer
from .functions import activation
from .functions import loss
from . import optimizer
from .models import Model

setup_variable()