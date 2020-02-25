#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage: input layer has 2 neurons, hidden layer has 2 neurons, output layer has 1 neuron
	NeuralNetwork nn({
		{2},
		{2, ActivationFunctionFactory::BuildActivationFunction(ActivationFunctions::Type::SIGMOID) },
		{1, ActivationFunctionFactory::BuildActivationFunction(ActivationFunctions::Type::SIGMOID) }
	});
	nn.SGD(1000, 0.01, { {1, 0}, {0, 1}, {0, 0}, {1, 1} }, { 1, 1, 0, 0 });
	for (unsigned int i = 0; i < 15; ++i)
	{
		auto output = nn.Predict({ 1, 0 });
		std::cout << "Prediction ->  Max value: " << output.value << ", Neuron# : " << output.index << std::endl;
	}
	std::cin.get();
	return 0;
}