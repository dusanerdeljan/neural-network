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

from pynn.activations import activation_functions
from pynn.validation import validate_activation_function, validate_neurons, validate_input
import ctypes as C


class Dense(C.Structure):
    _fields_ = [
        ("neurons", C.c_uint),
        ("activation_function", C.c_uint),
        ("inputs", C.c_uint)
    ]

    def __init__(self, neurons: int, activation: str, inputs: int = 0):
        super(Dense, self).__init__()
        validate_neurons(neurons)
        validate_input(inputs)
        validate_activation_function(activation)
        self.neurons = C.c_uint(neurons)
        self.activation_function = activation_functions[activation]
        self.inputs = C.c_uint(inputs)

    def __repr__(self):
        return f"<Dense: neurons={self.neurons}, activation={self.activation_function}, inputs={self.inputs}>"
