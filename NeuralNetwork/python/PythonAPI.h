#pragma once
#include "../NeuralNetwork.h"

#define NN_API extern "C" __declspec(dllexport)

typedef struct output
{
	double value;
	unsigned int argmax;
} Output;

typedef struct dense
{
	unsigned int neurons;
	unsigned int activation_function;
} Dense;

static nn::NeuralNetwork model(2, {
	nn::Layer(2, 4, nn::activation::SIGMOID),
	nn::Layer(4, 4, nn::activation::SIGMOID),
	nn::Layer(4, 1, nn::activation::SIGMOID)
}, nn::initialization::XAVIER_NORMAL, nn::loss::QUADRATIC);

NN_API void add(Dense d)
{
	std::cout << "Dense layer: " <<  d.neurons << ", " << d.activation_function << std::endl;
}

NN_API void train()
{
	std::vector<nn::TrainingData> trainingData({ { { 1, 0 }, 1 },{ { 1, 1 }, 0 },{ { 0, 1 }, 1 },{ { 0, 0 }, 0 } });
	unsigned int epochs = 1000;
	unsigned int batchSize = 1;
	double learningRate = 0.01;
	model.Train(nn::optimizer::Adam(learningRate), epochs, trainingData, batchSize, nn::regularizer::NONE);
}

NN_API Output eval(double inputs[])
{
	nn::Output out = model.Eval(std::vector<double>(inputs, inputs + 2));
	Output o; o.value = out.value; o.argmax = out.index;
	return o;
}


