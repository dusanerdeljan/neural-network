# neural-network
Simple neural network library written in C++ from scratch.

## Features

Currently supported featuers (library is still in it's very early phase):

### Optimizers

 * Gradient Descent
 * Gradient Descent with Momentum
 * Gradient Descent with Nestorov Momentum
 * Adagrad
 * Adam
 * Adadelta
 * RMSProp

### Activation functions

 * Sigmoid
 * ReLU
 * Leaky ReLU
 * ELU
 * Tanh
 
### Weight initializers

 * Random
 * Xavier Uniform
 * Xavier Normal
 * LeCun Uniform
 * LeCun Normal
 * He Uniform
 * He Normal
 
## Example usage

```cpp
// Training neural network to do XOR operation
NeuralNetwork model(2, {                          // Net has 2 inputs
		Layer(2, 4, Activation::Type::SIGMOID),       // First hidden layer
		Layer(4, 4, Activation::Type::SIGMOID),       // Second hidden layer
		Layer(4, 1, Activation::Type::SIGMOID)        // Output layer, net has 1 output
	}, new Initialization::XavierNormal());         // Weight initializer

// Getting the data
std::vector<NeuralNetwork::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });

// Training
unsigned int epochs = 5000;
double learningRate = 0.01;
model.Train(Optimizer::Type::RMSPROP, epochs, learningRate, trainingData);

// Evaluation
NeuralNetwork::Output res = model.Eval({ 0, 1 });
std::cout << "0 XOR 1 = " << res.value << std::endl;
std::cout << "Activated neuron index: " << res.index << std::endl;
```
