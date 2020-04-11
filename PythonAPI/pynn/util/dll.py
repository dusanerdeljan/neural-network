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
