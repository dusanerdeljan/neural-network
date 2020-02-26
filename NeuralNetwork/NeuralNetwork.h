#pragma once
#include <vector>
#include "Layer.h"

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

public:

	NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers);
	Output Predict(const std::vector<double>& input);
	~NeuralNetwork();
	void SGD(const int epochs, double lr, const std::vector<NeuralNetwork::TrainingData>& trainingData);
private:
	Matrix FeedForward(const std::vector<double>& input);
	Matrix MeanAbsoluteError(const NeuralNetwork::TrainingData& trainData);
	Matrix MeanSquaredError(const NeuralNetwork::TrainingData& trainData);
	std::pair<std::vector<Matrix>, std::vector<Matrix>> BackProp(const TrainingData& trainData) const;
	void InitializeProxies(std::vector<Matrix>& weightProxies, std::vector<Matrix>& biasProxies) const;
};

