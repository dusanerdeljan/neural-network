#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage: input layer has 2 neurons, output layer has 1 neuron
	NeuralNetwork nn(2, {	
		Layer(2, 4, ActivationFunctions::Type::SIGMOID),
		Layer(4, 1, ActivationFunctions::Type::SIGMOID)
	});

	// SGD training
	nn.SGD(10000, 0.1, { { { 1, 0 }, 1 }, { { 1, 1 }, 0 }, { { 0, 1 }, 1 }, { { 0, 0 }, 0 } });

	auto res = nn.Predict({ 0, 1 });
	std::cout << "0 XOR 1 = " << res.value << std::endl;

	auto res1 = nn.Predict({ 1, 0 });
	std::cout << "1 XOR 0 = " << res1.value << std::endl;

	auto res2 = nn.Predict({ 0, 0 });
	std::cout << "0 XOR 0 = " << res2.value << std::endl;

	auto res3 = nn.Predict({ 1, 1 });
	std::cout << "1 XOR 1 = " << res3.value << std::endl;

	std::cin.get();
	return 0;
}