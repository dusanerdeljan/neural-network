import ctypes as C


class Output(C.Structure):
    _fields_ = [
                ("value", C.c_double),
                ("argmax", C.c_int)
            ]