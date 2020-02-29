#pragma once
#include <vector>
#include "Layer.h"
#include "Optimizers.h"
#include "WeightInitializers.h"

class NeuralNetwork
{
public:
	struct Output
	{
		double value;
		unsigned int index;
		Output(double v, unsigned int i) : value(v), index(i) {}
	};

	struct TrainingData
	{
		std::vector<double> inputs;
		double target;
		TrainingData(const std::vector<double>& inputs, double target) : inputs(inputs), target(target) {}
		TrainingData& operator=(const NeuralNetwork::TrainingData& data)
		{
			inputs = data.inputs;
			target = data.target;
			return *this;
		}
	};

private:
	unsigned int m_InputSize;
	std::vector<Layer> m_Layers;
	Initialization::Initializer* m_WeightInitializer;

public:
	NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers, Initialization::Initializer* initializer);
	NeuralNetwork(NeuralNetwork&& net);
	void Train(Optimizer::Type optimizer, unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize=1);
	Output Eval(const std::vector<double>& input);
	void SaveModel(const char* fileName) const;
	static NeuralNetwork LoadModel(const char* fileName);
	~NeuralNetwork();
private:
	Matrix FeedForward(const std::vector<double>& input);
	Matrix MeanAbsoluteError(const NeuralNetwork::TrainingData& trainData);
	Matrix MeanSquaredError(const NeuralNetwork::TrainingData& trainData);

	// Optimizers
	void SGD(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);
	void Momentum(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);
	void Nesterov(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);
	void Adagrad(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);
	void RMSprop(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);
	void Adadelta(unsigned int epochs, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);
	void Adam(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize);



};

