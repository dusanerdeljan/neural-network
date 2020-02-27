#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage: input layer has 2 neurons, output layer has 1 neuron
	NeuralNetwork nn(2, {	
		Layer(2, 1, ActivationFunctions::Type::SIGMOID),			// Input
	});

	// Simple training
	nn.SimpleTraining(10000, { {{1, 0}, 1},{ { 1, 1 }, 0},{ { 0, 1 }, 0},{ { 0, 0 }, 0} });

	std::cin.get();
	return 0;
}