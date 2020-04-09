from activations import activation_functions
import ctypes as C


class Dense(C.Structure):
    _fields_ = [
        ("neurons", C.c_uint),
        ("activation_function", C.c_uint),
        ("inputs", C.c_uint)
    ]

    def __init__(self, neurons: C.c_uint, activation: str, inputs: C.c_uint = 0):
        super(Dense, self).__init__()
        self.neurons = neurons
        if activation in activation_functions:
            self.activation_function = activation_functions[activation]
        else:
            raise Exception("Invalid activation functions.\n"
                            "Available activation functions are:\n"
                            "\tsigmoid\n"
                            "\trelu\n"
                            "\tleaky_relu\n"
                            "\telu\n"
                            "\ttanh\n"
                            "\tsoftmax")
        self.inputs = inputs