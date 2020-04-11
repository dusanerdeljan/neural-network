from pynn.optimizers import get_optimizer


class State:

    def __init__(self):
        self._optimizer = None
        self._regularizer = None
        self._initializer = None
        self._loss = None
        self._input_size = 0
        self._output_size = 0
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def update(self, optimizer, loss, initializer, regularizer, input_size, output_size):
        self._optimizer = get_optimizer(optimizer)
        self._loss = loss
        self._initializer = initializer
        self._regularizer = regularizer
        self._input_size = input_size
        self._output_size = output_size