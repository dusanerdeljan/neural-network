"""
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""

from pynn.util.dll import DLLUtil
from pynn.layers.dense import Dense
from pynn.type.output import Output
from pynn import optimizers, losses, regularizers, weightinitializers
from pynn.validation import validate_compile, validate_fit
from pynn.state import State
import numpy as np
import ctypes as C
import pickle
import os


class NeuralNetwork(object):

    def __init__(self, layers=None):
        self._lib = DLLUtil.load_dll()
        self._layers = None
        self._compiled = False
        self._state = State()
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
        self._state.add_layer(layer)
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

    def _update_state(self, optimizer, loss, initializer, regularizer):
        self._state.update(optimizer, loss, initializer, regularizer, self._layers[0].inputs, self._layers[-1].neurons)

    def compile(self, optimizer='sgd', loss='mean_squared_error', initializer='random', regularizer='none'):
        if not self._layers:
            raise Exception("No layers specified.")
        validate_compile(optimizer, loss, initializer, regularizer)
        for layer in self._layers:
            self._lib.add(layer)
        self._update_state(optimizer, loss, initializer, regularizer)
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
        if not self._compiled:
            raise Exception("Model is not compiled.")
        self._lib.save(C.c_char_p(bytes(os.path.abspath(file_path), encoding='utf-8')))
        state_file_path = os.path.join(os.path.dirname(file_path), ".state." + os.path.basename(file_path))
        with open(state_file_path, 'wb') as state_file:
            pickle.dump(self._state, state_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path: str):
        state_file_path = os.path.join(os.path.dirname(file_path), ".state." + os.path.basename(file_path))
        if not (os.path.exists(file_path) or os.path.exists(state_file_path)):
            raise Exception("Invalid model file path.")
        net = NeuralNetwork()
        net._lib.load(C.c_char_p(bytes(os.path.abspath(file_path), encoding='utf-8')))
        with open(state_file_path, 'rb') as file:
            net._state = pickle.load(file)
        net._compiled = True
        net._lib.state_loaded(C.byref(net._state._optimizer),
                              optimizers.optimizers[net._state._optimizer.__class__.__name__.lower()],
                              regularizers.regularizers[net._state._regularizer], C.c_uint(net._state._input_size),
                              C.c_uint(net._state._output_size))
        return net


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
    model.save('model.bin')
    # model = NeuralNetwork.load('model.bin')
    # evaluate(model)
