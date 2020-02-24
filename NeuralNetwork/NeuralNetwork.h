#pragma once
#include <vector>
#include "Matrix.h"

class NeuralNetwork
{
private:
	std::vector<Matrix> m_WeightMatrices;
	std::vector<Matrix> m_Biases;
public:
	NeuralNetwork(const std::vector<unsigned int>& layerNeurons);
	double Predict(const std::vector<double>& input) const;
	~NeuralNetwork();
};

