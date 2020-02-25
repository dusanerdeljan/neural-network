#pragma once
#include <vector>
#include "Matrix.h"
#include "ActivationFunctions.h"

class NeuralNetwork
{
public:
	struct Output
	{
		double value;
		unsigned int index;
		Output(double v, unsigned int i) : value(v), index(i) {}
	};

	struct LayerOptions
	{
		unsigned int neuronCount;
		ActivationFunctions::ActivationFunction* activationFunction;
		LayerOptions(unsigned int count, ActivationFunctions::ActivationFunction* func=nullptr) : neuronCount(count), activationFunction(func) {}
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
	std::vector<Matrix> m_WeightMatrices;
	std::vector<Matrix> m_Biases;
	std::vector<LayerOptions> m_LayerOptions;

public:

	NeuralNetwork(const std::vector<NeuralNetwork::LayerOptions>& layerOptions);
	Output Predict(const std::vector<double>& input) const;
	~NeuralNetwork();
	void SGD(const int epochs, double lr, const std::vector<NeuralNetwork::TrainingData>& trainingData);
private:
	Matrix FeedForward(const std::vector<double>& input) const;
	std::vector<Matrix> TrainFeedForward(const std::vector<double>& input) const;
	Matrix MeanAbsoluteError(const NeuralNetwork::TrainingData& trainData);
	Matrix MeanSquaredError(const NeuralNetwork::TrainingData& trainData);
	std::pair<std::vector<Matrix>, std::vector<Matrix>> BackProp(const TrainingData& trainData) const;
	void InitializeProxies(std::vector<Matrix>& weightProxies, std::vector<Matrix>& biasProxies) const;
};

