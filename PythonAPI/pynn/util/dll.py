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

from pynn.type.output import Output
from pynn.layers.dense import Dense
import ctypes as C
import numpy as np
import os


class DLLUtil(object):

    @staticmethod
    def load_dll() -> C.CDLL:
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../lib/NeuralNetwork.dll")
        if not os.path.exists(path):
            raise Exception("Invalid DLL path. DLL is missing.")
        library = C.cdll.LoadLibrary(path)
        DLLUtil._configure(library)
        return library

    @staticmethod
    def _configure(library: C.CDLL):
        library.eval.argtypes = [np.ctypeslib.ndpointer(dtype=np.double)]
        library.eval.restype = Output
        library.add.argtypes = [C.POINTER(Dense)]
        library.add_training_sample.argtypes = [np.ctypeslib.ndpointer(dtype=np.double),
                                                np.ctypeslib.ndpointer(dtype=np.double)]
        library.compile.argtypes = [C.c_uint, C.c_uint, C.c_uint, C.c_uint]
        library.train.argtypes = [C.c_uint, C.c_uint]
        library.compile_optimizer.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint]
        library.save.argtypes = [C.c_char_p]
        library.load.argtypes = [C.c_char_p]
        library.state_loaded.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint]
