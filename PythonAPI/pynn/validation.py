from pynn.activations import activation_functions
from pynn.optimizers import optimizers, Optimizer
from pynn.losses import losses
from pynn.weightinitializers import weight_initializers
from pynn.regularizers import regularizers


def validate_neurons(neurons: int):
    if not isinstance(neurons, int):
        raise Exception("Invalid type. Expected 'int'.")
    if neurons <= 0:
        raise Exception("Numbers of neurons can't be <= 0.")


def validate_input(inputs: int):
    if not isinstance(inputs, int):
        raise Exception("Invalid type. Expected 'int'.")
    if inputs < 0:
        raise Exception("Number of inputs can't be < 0.")


def validate_activation_function(activation: str):
    if activation not in activation_functions:
        raise Exception("""Invalid activation functions.
Available activation functions are: sigmoid, relu, leaky_relu, elu, tanh, softmax
        """)


def validate_fit(epochs: int, batch_size: int):
    if not isinstance(epochs, int):
        raise Exception("Invalid epoch type. Expected 'int'.")
    if not isinstance(batch_size, int):
        raise Exception("Invalid batch size type. Expected 'int'.")
    if epochs <= 0:
        raise Exception("Number of epochs can't be <= 0.")
    if batch_size <= 0:
        raise Exception("Batch size can't be <= 0.")


def validate_compile(optimizer, loss: str, initializer: str, regularizer: str):
    validate_optimizer(optimizer)
    validate_loss(loss)
    validate_initializer(initializer)
    validate_regularizer(regularizer)


def validate_optimizer(optimizer):
    if isinstance(optimizer, str):
        if optimizer not in optimizers:
            raise Exception("""Invalid optimizer type.
Available optimizers: sgd, momentum, nesterov, rmsprop, adam, nadam, adadelta, adagrad, adamax, amsgrad""")
    elif not isinstance(optimizer, Optimizer):
        raise Exception("Invalid optimizer type. Expected 'str' or 'Optimizer'.")


def validate_loss(loss: str):
    if isinstance(loss, str):
        if loss not in losses:
            raise Exception("""Invalid loss type.
Available losses: mean_absolute_error, mean_squared_error, quadratic, half_quadratic, cross_entropy, nll""")
    else:
        raise Exception("Invalid loss type. Expected 'str'.")


def validate_initializer(initializer: str):
    if isinstance(initializer, str):
        if initializer not in weight_initializers:
            raise Exception("""Invalid weight initializer type.
Available weight initializers: random, xavier_normal, xaiver_uniform, he_normal, he_uniform, lecun_normal, lecun_uniform""")
    else:
        raise Exception("Invalid loss type. Expected 'str'.")


def validate_regularizer(regularizer: str):
    if isinstance(regularizer, str):
        if regularizer not in regularizers:
            raise Exception("""Invalid regularizer type.
Available regularizers: none, l1, l2, l1l2""")
    else:
        raise Exception("Invalid loss type. Expected 'str'.")
