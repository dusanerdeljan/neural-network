from util.dll import DLLUtil
from layers.dense import Dense
from type.output import Output
import numpy as np


class NeuralNetwork(object):
    
    def __init__(self):
        self._lib = DLLUtil.load_dll()

    def add(self, layer):
        self._lib.add(layer)

    def train(self):
        self._lib.train()

    def eval(self, inputs):
        return self._lib.eval(np.asarray(inputs, dtype=np.double))


def evaluate(model):
    output: Output = model.eval([0, 1])
    print(f"0 XOR 1 = {output.value}")

    output: Output = model.eval([0, 0])
    print(f"0 XOR 0 = {output.value}")

    output: Output = model.eval([1, 0])
    print(f"1 XOR 0 = {output.value}")

    output: Output = model.eval([1, 1])
    print(f"1 XOR 1 = {output.value}")


if __name__ == "__main__":
    model = NeuralNetwork()
    model.train()
    evaluate(model)
    model.add(Dense(50, 'relu'))
