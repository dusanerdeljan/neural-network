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
