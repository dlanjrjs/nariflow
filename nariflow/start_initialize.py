import os

class start_initializer():
    def __init__(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        self.path = os.path.join(my_path, "./conf.txt")

    def initializer(self, where):
        if where == 'node':
            with open(self.path, mode = 'w') as f:
                f.write('node')
                return

        if where == 'tape':
            with open(self.path, mode = 'w') as f:
                f.write('tape')
                return

        else :
            raise Exception('You should choose within two mods, node or tape')



