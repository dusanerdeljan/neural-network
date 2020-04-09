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
	unsigned int inputs;
} Dense;

static std::unique_ptr<nn::NeuralNetwork> model;

static std::vector<nn::TrainingData> trainingData;
static std::vector<nn::Layer> layers;
static unsigned int optimizerType;

NN_API void compile(unsigned int optimizer, unsigned int loss, unsigned int initializer, unsigned int regularizer)
{
	optimizer = optimizerType;
	// TODO: input hardcoded to 2
	model = std::make_unique<nn::NeuralNetwork>(nn::NeuralNetwork(2, std::move(layers), nn::initialization::Type(initializer), nn::loss::Type(loss)));
	std::cout << "Compiled model succesfully!" << std::endl;
}

NN_API void add(Dense* dense)
{
	layers.emplace_back(dense->inputs, dense->neurons, nn::activation::Type(dense->activation_function));
	std::cout << layers.size() << std::endl;
}

NN_API void add_training_sample(double inputs[], double targets[], unsigned int inputDim, unsigned int outputDim)
{
	trainingData.emplace_back(std::vector<double>(inputs, inputs+inputDim), std::vector<double>(targets, targets+outputDim));
}

NN_API void train(double learningRate, unsigned int epochs, unsigned int batchSize)
{
	// TODO: optimizer and regularizer hardcoded
	model->Train(nn::optimizer::Adam(learningRate), epochs, trainingData, batchSize, nn::regularizer::NONE);
	trainingData.clear();
}

NN_API Output eval(double inputs[])
{
	// TODO: input size hardcoded to 2
	nn::Output out = model->Eval(std::vector<double>(inputs, inputs + 2));
	Output o; o.value = out.value; o.argmax = out.index;
	return o;
}


