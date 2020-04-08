from layers.layer import activation_functions
import ctypes as C


class Dense(C.Structure):
    _fields_ = [
        ("neurons", C.c_uint),
        ("activation_function", C.c_uint)
    ]

    def __init__(self, n: C.c_uint, af: str):
        self.neurons = n
        self.activation_function = activation_functions[af]