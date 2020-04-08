from type.output import Output
from layers.dense import Dense
import ctypes as C
import numpy as np

class DLLUtil(object):

    @staticmethod
    def load_dll():
        library = C.cdll.LoadLibrary('E:\\Dev\\DeepLearningLibrary\\NeuralNetwork\\PythonAPI\\lib\\NeuralNetwork.dll')
        DLLUtil._configure(library)
        return library

    @staticmethod
    def _configure(library):
        library.eval.argtypes = [np.ctypeslib.ndpointer(dtype=np.double)]
        library.eval.restype = Output
        library.add.argypes = [Dense]