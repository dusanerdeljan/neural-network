#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage: input layer has 2 neurons, hidden layer has 2 neurons, output layer has 1 neuron
	NeuralNetwork nn({ 2, 2, 1 });
	auto output = nn.Predict({ 0, 1 });
	std::cout << "Prediction ->  Max value: " << output.value << ", Neuron# : " << output.index <<  std::endl;
	std::cin.get();
	return 0;
}