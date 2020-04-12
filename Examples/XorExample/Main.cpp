#include <iostream>
#include "NeuralNetwork.h"

int main()
{
	// Example usage
	// TODO: 'nn' namespace sam stavio kao placeholder dok ne smislimo nesto bolje
	nn::NeuralNetwork model(2, {
		nn::Layer(2, 4, nn::activation::SIGMOID),
		nn::Layer(4, 4, nn::activation::SIGMOID),
		nn::Layer(4, 1, nn::activation::SIGMOID)
	}, nn::initialization::XAVIER_NORMAL, nn::loss::QUADRATIC);

	// Getting the data
	std::vector<nn::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });

	// Training
	unsigned int epochs = 1000;
	unsigned int batchSize = 1; // Na ovom primeru bas i nema nekog smisla batch i regularizer, ali testirao sam na MNIST datasetu lepo je radilo
	double learningRate = 0.01;
	model.Train(nn::optimizer::Adam(learningRate), epochs, trainingData, batchSize, nn::regularizer::NONE);

	//model.SaveModel("model.bin");
	//nn::NeuralNetwork model = nn::NeuralNetwork::LoadModel("model.bin");
	// Evaluation
	auto res = model.Eval({ 0, 1 });
	std::cout << "0 XOR 1 = " << res.Value << std::endl;

	auto res1 = model.Eval({ 1, 0 });
	std::cout << "1 XOR 0 = " << res1.Value << std::endl;

	auto res2 = model.Eval({ 0, 0 });
	std::cout << "0 XOR 0 = " << res2.Value << std::endl;

	auto res3 = model.Eval({ 1, 1 });
	std::cout << "1 XOR 1 = " << res3.Value << std::endl;

	std::cin.get();
	return 0;
}