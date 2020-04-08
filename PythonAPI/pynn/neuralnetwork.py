from util.dll import DLLUtil
from layers.dense import Dense


class NeuralNetwork(object):
    
    def __init__(self):
        self._lib = DLLUtil.load_dll()

    def add(self, layer):
        self._lib.add(layer)

    def train(self):
        self._lib.train()


if __name__ == "__main__":
    model = NeuralNetwork()
    model.train()
    model.add(Dense(50, 'relu'))
