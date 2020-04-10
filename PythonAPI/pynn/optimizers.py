import ctypes as C

optimizers = {
    'sgd': 0,
    'momentum': 1,
    'nesterov': 2,
    'adagrad': 3,
    'rmsprop': 4,
    'adadelta': 5,
    'adam': 6,
    'nadam': 7,
    'adamax': 8,
    'amsgrad': 9
}

default_lr = 0.01


class Optimizer(C.Structure):
    pass


class SGD(Optimizer):
    _fields_ = [
        ('lr', C.c_double)
    ]

    def __init__(self, lr=default_lr):
        self.lr = C.c_double(lr)


class Momentum(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('momentum', C.c_double)
    ]

    def __init__(self, lr=default_lr, momentum=0.9):
        self.lr = C.c_double(lr)
        self.momentum = C.c_double(momentum)


class Nesterov(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('momentum', C.c_double)
    ]

    def __init__(self, lr=default_lr, momentum=0.9):
        self.lr = C.c_double(lr)
        self.momentum = C.c_double(momentum)


class Adagrad(Optimizer):
    _fields_ = [
        ('lr', C.c_double)
    ]

    def __init__(self, lr=default_lr):
        self.lr = C.c_double(lr)


class RMSProp(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta=0.999):
        self.lr = C.c_double(lr)
        self.beta = C.c_double(beta)


class Adadelta(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta=0.999):
        self.lr = C.c_double(lr)
        self.beta = C.c_double(beta)


class Adam(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999):
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)


class Nadam(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999):
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)


class Adamax(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999):
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)


class AMSGrad(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999):
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)