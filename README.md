# neural-network
Simple neural network library written in C++ from scratch.

## Features

Currently supported featuers (library is still in it's very early phase):

### Optimizers

 * Gradient Descent
 * Gradient Descent with Momentum
 * Gradient Descent with Nesterov Momentum
 * Adagrad
 * Adam
 * Nadam
 * Adadelta
 * RMSProp

### Activation functions

 * Sigmoid
 * ReLU
 * Leaky ReLU
 * ELU
 * Tanh
 * Softmax

### Loss functions

 * Mean Absolute Error
 * Mean Squared Error
 * Quadratic
 * Half Quadratic
 * Cross Entropy
 * NLL
 
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
// Example usage
NeuralNetwork model(2, {	// Net has 2 inputs
	Layer(2, 4, new Activation::Sigmoid()),	// First hidden layer
	Layer(4, 4, new Activation::Sigmoid()),	// Second hidden layer
	Layer(4, 1, new Activation::Sigmoid())	// Output layer, net has 1 output
}, new Initialization::XavierNormal(), new Loss::Quadratic());	// Weight initializer, loss function

// Getting the data
std::vector<NeuralNetwork::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });

// Training
unsigned int epochs = 1000;
double learningRate = 0.01;
model.Train(new Optimizer::Adam(learningRate), epochs, trainingData);

model.SaveModel("model.bin");
//NeuralNetwork model = NeuralNetwork::LoadModel("model.bin");
// Evaluation
NeuralNetwork::Output res = model.Eval({ 0, 1 }); // Alternatively auto res = model({0, 1});
std::cout << "0 XOR 1 = " << res.value << std::endl;
std::cout << "Activated neuron index: " << res.index << std::endl;
```
