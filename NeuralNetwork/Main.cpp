#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage
	NeuralNetwork model(2, {	
		Layer(2, 4, new Activation::Sigmoid()),
		Layer(4, 4, new Activation::Sigmoid()),
		Layer(4, 1, new Activation::Sigmoid())
	}, new Initialization::XavierNormal(), new Loss::MeanSquaredError());

	// Getting the data
	std::vector<NeuralNetwork::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });

	// Training
	unsigned int epochs = 1000;
	double learningRate = 0.01;
	model.Train(new Optimizer::Adam(learningRate), epochs, learningRate, trainingData);

	//model.SaveModel("model.bin");
	//NeuralNetwork model = NeuralNetwork::LoadModel("model.bin");
	// Evaluation
	auto res = model.Eval({ 0, 1 });
	std::cout << "0 XOR 1 = " << res.value << std::endl;

	auto res1 = model.Eval({ 1, 0 });
	std::cout << "1 XOR 0 = " << res1.value << std::endl;

	auto res2 = model.Eval({ 0, 0 });
	std::cout << "0 XOR 0 = " << res2.value << std::endl;

	auto res3 = model.Eval({ 1, 1 });
	std::cout << "1 XOR 1 = " << res3.value << std::endl;

	std::cin.get();
	return 0;
}