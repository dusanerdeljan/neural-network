from pynn.util.dll import DLLUtil
from pynn.layers.dense import Dense
from pynn.type.output import Output
from pynn import optimizers, losses, regularizers, weightinitializers
from pynn.validation import validate_compile, validate_fit
import numpy as np
import ctypes as C


class NeuralNetwork(object):

    def __init__(self, layers=None):
        self._lib = DLLUtil.load_dll()
        self._layers = None
        self._compiled = False
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Dense):
        if not self._layers:
            if layer.inputs <= 0:
                raise Exception("Invalid layer inputs.")
            self._layers = [layer]
        elif self._layers[-1].neurons != layer.inputs and layer.inputs != 0:
            raise Exception("Invalid layer inputs.")
        else:
            layer.inputs = self._layers[-1].neurons
            self._layers.append(layer)
        self._compiled = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int = 1):
        if not self._compiled:
            raise Exception("Model is not compiled.")
        validate_fit(epochs, batch_size)
        for x, y in zip(x_train, y_train):
            self._lib.add_training_sample(np.asarray(x, dtype=np.double),
                                          np.asarray(y if np.isscalar(y) else [y], dtype=np.double))
        self._lib.train(C.c_uint(epochs), C.c_uint(batch_size))

    def predict(self, inputs: np.array) -> Output:
        return self._lib.eval(np.asarray(inputs, dtype=np.double))

    def compile(self, optimizer='sgd', loss='mean_squared_error', initializer='random', regularizer='none'):
        if not self._layers:
            raise Exception("No layers specified.")
        validate_compile(optimizer, loss, initializer, regularizer)
        for layer in self._layers:
            self._lib.add(layer)
        if isinstance(optimizer, str):
            self._lib.compile(optimizers.optimizers[optimizer], losses.losses[loss],
                              weightinitializers.weight_initializers[initializer],
                              regularizers.regularizers[regularizer])
        else:
            self._lib.compile_optimizer(C.byref(optimizer), optimizers.optimizers[optimizer.__class__.__name__.lower()],
                                        losses.losses[loss],
                                        weightinitializers.weight_initializers[initializer],
                                        regularizers.regularizers[regularizer])
        self._compiled = True

    def save(self, file_path: str):
        pass

    @staticmethod
    def load(file_path: str):
        pass


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
    x = [[0, 1], [0, 0], [1, 0], [1, 1]]
    y = [[1], [0], [1], [0]]
    model = NeuralNetwork([
        Dense(4, 'sigmoid', inputs=2),
        Dense(4, 'sigmoid'),
        Dense(1, 'sigmoid')
    ])
    # model.add(Dense(4, 'sigmoid', inputs=2))
    # model.add(Dense(4, 'sigmoid'))
    # model.add(Dense(1, 'sigmoid'))
    model.compile(optimizer='adam', loss='quadratic',
                  initializer='xavier_normal', regularizer='none')
    model.fit(np.array(x), np.array(y), epochs=1000, batch_size=1)
    evaluate(model)
