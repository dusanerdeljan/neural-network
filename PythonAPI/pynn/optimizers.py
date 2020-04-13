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
    'amsgrad': 9,
    'adabound': 10,
    'amsbound': 11
}

default_lr = 0.01


def get_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer
    if optimizer == 'sgd':
        return SGD()
    elif optimizer == 'momentum':
        return Momentum()
    elif optimizer == 'nesterov':
        return Nesterov()
    elif optimizer == 'adagrad':
        return Adagrad()
    elif optimizer == 'adadelta':
        return Adadelta()
    elif optimizer == 'adam':
        return Adam()
    elif optimizer == 'nadam':
        return Nadam()
    elif optimizer == 'adamax':
        return Adamax()
    elif optimizer == 'amsgrad':
        return AMSGrad()


class Optimizer(C.Structure):
    pass


class SGD(Optimizer):
    _fields_ = [
        ('lr', C.c_double)
    ]

    def __init__(self, lr=default_lr):
        super(SGD, self).__init__()
        self.lr = C.c_double(lr)


class Momentum(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('momentum', C.c_double)
    ]

    def __init__(self, lr=default_lr, momentum=0.9):
        super(Momentum, self).__init__()
        self.lr = C.c_double(lr)
        self.momentum = C.c_double(momentum)


class Nesterov(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('momentum', C.c_double)
    ]

    def __init__(self, lr=default_lr, momentum=0.9):
        super(Nesterov, self).__init__()
        self.lr = C.c_double(lr)
        self.momentum = C.c_double(momentum)


class Adagrad(Optimizer):
    _fields_ = [
        ('lr', C.c_double)
    ]

    def __init__(self, lr=default_lr):
        super(Adagrad, self).__init__()
        self.lr = C.c_double(lr)


class RMSProp(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta=0.999):
        super(RMSProp, self).__init__()
        self.lr = C.c_double(lr)
        self.beta = C.c_double(beta)


class Adadelta(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta=0.999):
        super(Adadelta, self).__init__()
        self.lr = C.c_double(lr)
        self.beta = C.c_double(beta)


class Adam(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__()
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
        super(Nadam, self).__init__()
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
        super(Adamax, self).__init__()
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
        super(AMSGrad, self).__init__()
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)


class Adabound(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double),
        ('final_lr', C.c_double),
        ('gamma', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=1e-3):
        super(Adabound, self).__init__()
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)
        self.final_lr = C.c_double(final_lr)
        self.gamma = C.c_double(gamma)


class AMSBound(Optimizer):
    _fields_ = [
        ('lr', C.c_double),
        ('beta1', C.c_double),
        ('beta2', C.c_double),
        ('final_lr', C.c_double),
        ('gamma', C.c_double)
    ]

    def __init__(self, lr=default_lr, beta1=0.9, beta2=0.999, final_lr=0.1, gamma=1e-3):
        super(AMSBound, self).__init__()
        self.lr = C.c_double(lr)
        self.beta1 = C.c_double(beta1)
        self.beta1 = C.c_double(beta2)
        self.final_lr = C.c_double(final_lr)
        self.gamma = C.c_double(gamma)
