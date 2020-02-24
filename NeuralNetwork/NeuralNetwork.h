#pragma once
#include <vector>
#include "Matrix.h"

class NeuralNetwork
{
private:
	std::vector<Matrix> m_WeightMatrices;
	std::vector<Matrix> m_Biases;
public:
	struct Output
	{
		double value;
		unsigned int index;
		Output(double v, unsigned int i) : value(v), index(i) {}
	};

	NeuralNetwork(const std::vector<unsigned int>& layerNeurons);
	Output Predict(const std::vector<double>& input) const;
	~NeuralNetwork();
	auto gradientDescent(const int epochs, double lr, const int batchSize = 0);

private:
	std::vector<double> FeedForward(const std::vector<double>& input) const;
};

