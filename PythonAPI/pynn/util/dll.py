from type.output import Output
from layers.dense import Dense
import ctypes as C
import numpy as np
import os


class DLLUtil(object):

    @staticmethod
    def load_dll():
        library = C.cdll.LoadLibrary(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../lib/NeuralNetwork.dll"))
        DLLUtil._configure(library)
        return library

    @staticmethod
    def _configure(library):
        library.eval.argtypes = [np.ctypeslib.ndpointer(dtype=np.double)]
        library.eval.restype = Output
        library.add.argtypes = [Dense]
