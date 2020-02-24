#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage: input layer has 2 neurons, hidden layer has 2 neurons, output layer has 1 neuron
	NeuralNetwork nn({ 2, 2, 1 });
	double result = nn.Predict({ 0, 1 });
	std::cin.get();
	return 0;
}