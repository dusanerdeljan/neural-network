# neural-network
Simple deep learning library written in C++ from scratch.

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
nn::NeuralNetwork model(2, {	
	nn::Layer(2, 4, nn::activation::SIGMOID),
	nn::Layer(4, 4, nn::activation::SIGMOID),
	nn::Layer(4, 1, nn::activation::SIGMOID)
}, nn::initialization::XAVIER_NORMAL, nn::loss::QUADRATIC);

// Getting the data
std::vector<nn::NeuralNetwork::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });

// Training
unsigned int epochs = 1000;
double learningRate = 0.01;
model.Train(nn::optimizer::Adam(learningRate), epochs, trainingData);

model.SaveModel("model.bin");
//nn::NeuralNetwork model = nn::NeuralNetwork::LoadModel("model.bin");
// Evaluation
nn::NeuralNetwork::Output res = model.Eval({ 0, 1 }); // Alternatively auto res = model({0, 1});
std::cout << "0 XOR 1 = " << res.value << std::endl;
std::cout << "Activated neuron index: " << res.index << std::endl;
```
