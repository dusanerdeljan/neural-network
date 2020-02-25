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

private:
	std::vector<Matrix> m_WeightMatrices;
	std::vector<Matrix> m_Biases;
	std::vector<LayerOptions> m_LayerOptions;

public:

	NeuralNetwork(const std::vector<NeuralNetwork::LayerOptions>& layerOptions);
	Output Predict(const std::vector<double>& input) const;
	~NeuralNetwork();
	void SGD(const int epochs, double lr, const std::vector<std::vector<double>>& inputs, const std::vector<double>& labels);
	Matrix MeanAbsoluteError(const std::vector<double>& input, double target);
	Matrix MeanSquaredError(const std::vector<double>& input, double target);
private:
	Matrix FeedForward(const std::vector<double>& input) const;
	std::vector<Matrix> TrainFeedForward(const std::vector<double>& input) const;
};

