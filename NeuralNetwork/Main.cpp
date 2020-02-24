#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// example usage
	// Input layer has 2 neurons
	// Hidden layer has 2 neurons
	// Output layer has 1 neuron
	NeuralNetwork nn({ 2, 2, 1 });
	double result = nn.Predict({ 0, 1 });
	std::cin.get();
	return 0;
}