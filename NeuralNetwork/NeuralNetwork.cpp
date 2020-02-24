#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<unsigned int>& layerNeurons) 
{
	for (unsigned int i = 0; i < layerNeurons.size() - 1; ++i)
	{
		// For example if the input layer has 5 neurons and the first hidden layer has 3 neurons, weight matrix is 3*5, and bias matrix is 3*1
		m_WeightMatrices.push_back(Matrix(layerNeurons[i + 1], layerNeurons[i]));
		m_Biases.push_back(Matrix(layerNeurons[i + 1], 1));
	}
}

double NeuralNetwork::Predict(const std::vector<double>& input) const
{
	return 0.0;
}

NeuralNetwork::~NeuralNetwork()
{
}
