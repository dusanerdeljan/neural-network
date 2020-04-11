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

from pynn.optimizers import get_optimizer


class State:

    def __init__(self):
        self._optimizer = None
        self._regularizer = None
        self._initializer = None
        self._loss = None
        self._input_size = 0
        self._output_size = 0
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def update(self, optimizer, loss, initializer, regularizer, input_size, output_size):
        self._optimizer = get_optimizer(optimizer)
        self._loss = loss
        self._initializer = initializer
        self._regularizer = regularizer
        self._input_size = input_size
        self._output_size = output_size