from activations import activation_functions
import ctypes as C


class Dense(C.Structure):
    _fields_ = [
        ("neurons", C.c_uint),
        ("activation_function", C.c_uint),
        ("inputs", C.c_uint)
    ]

    def __init__(self, neurons: int, activation: str, inputs: int = 0):
        super(Dense, self).__init__()
        self.neurons = C.c_uint(neurons)
        if activation in activation_functions:
            self.activation_function = activation_functions[activation]
        else:
            raise Exception("Invalid activation function.\n"
                            "Available activation functions are:\n"
                            "\tsigmoid\n"
                            "\trelu\n"
                            "\tleaky_relu\n"
                            "\telu\n"
                            "\ttanh\n"
                            "\tsoftmax")
        self.inputs = C.c_uint(inputs)

    def __repr__(self):
        return f"<Dense: neurons={self.neurons}, activation={self.activation_function}, inputs={self.inputs}>"
