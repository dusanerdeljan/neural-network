from util.dll import DLLUtil
from layers.dense import Dense
from type.output import Output
from optimizers import optimizers
from losses import losses
from regularizers import regularizers
from weightinitializers import weight_initializers
import numpy as np
import ctypes as C


class NeuralNetwork(object):
    
    def __init__(self):
        self._input_dim = 2
        self._output_dim = 1
        self._lib = DLLUtil.load_dll()
        self._layers = None
        self._compiled = False

    def add(self, layer: Dense):
        if not self._layers:
            if layer.inputs <= 0:
                raise Exception("Invalid layer inputs!")
            self._layers = [layer]
        elif self._layers[-1].neurons != layer.inputs and layer.inputs != 0:
            raise Exception("Invalid layer inputs!")
        else:
            layer.inputs = self._layers[-1].neurons
            self._layers.append(layer)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, lr, epochs, batch_size=1):
        if not self._compiled:
            raise Exception("Model is not compiled!")
        for x, y in zip(x_train, y_train):
            self._lib.add_training_sample(np.asarray(x, dtype=np.double),
                                          np.asarray(y if np.isscalar(y) else [y], dtype=np.double),
                                          self._input_dim, self._output_dim)
        self._lib.train(C.c_double(lr), C.c_uint(epochs), C.c_uint(batch_size))

    def predict(self, inputs: np.array) -> Output:
        return self._lib.eval(np.asarray(inputs, dtype=np.double))

    def compile(self, optimizer='sgd', loss='mean_squared_error', initializer='random', regularizer='none'):
        if not self._layers:
            raise Exception("No layers specified!")
        for layer in self._layers:
            self._lib.add(layer)
        self._lib.compile(optimizers[optimizer], losses[loss], weight_initializers[initializer], regularizers[regularizer])
        self._compiled = True


def evaluate(model):
    output: Output = model.predict(np.array([0, 1]))
    print(f"0 XOR 1 = {output.value}")

    output: Output = model.predict(np.array([0, 0]))
    print(f"0 XOR 0 = {output.value}")

    output: Output = model.predict(np.array([1, 0]))
    print(f"1 XOR 0 = {output.value}")

    output: Output = model.predict(np.array([1, 1]))
    print(f"1 XOR 1 = {output.value}")


if __name__ == "__main__":
    model = NeuralNetwork()
    x = [[0, 1], [0, 0], [1, 0], [1, 1]]
    y = [[1], [0], [1], [0]]
    model.add(Dense(4, 'sigmoid', inputs=2))
    model.add(Dense(4, 'sigmoid'))
    model.add(Dense(1, 'sigmoid'))
    model.compile('adam', 'quadratic', 'xavier_normal', 'none')
    model.fit(np.array(x), np.array(y), lr=0.01, epochs=1000, batch_size=1)
    evaluate(model)