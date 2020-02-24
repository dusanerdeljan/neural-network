#include "NeuralNetwork.h"
#include <algorithm>
#include <functional>

NeuralNetwork::NeuralNetwork(const std::vector<NeuralNetwork::LayerOptions>& layerOptions)
{
	for (unsigned int i = 0; i < layerOptions.size() - 1; ++i)
	{
		// For example if the input layer has 5 neurons and the first hidden layer has 3 neurons, weight matrix is 3*5, and bias matrix is 3*1
		m_WeightMatrices.push_back(Matrix(layerOptions[i + 1].neuronCount, layerOptions[i].neuronCount));
		m_Biases.push_back(Matrix(layerOptions[i + 1].neuronCount, 1));
		m_LayerOptions.push_back(layerOptions[i + 1]); // Ignore the input layer
	}
}

NeuralNetwork::Output NeuralNetwork::Predict(const std::vector<double>& input) const
{
	std::vector<double> outputResults = FeedForward(input);
	unsigned int maxIndex = std::max_element(outputResults.begin(), outputResults.end()) - outputResults.begin();
	return{ outputResults[maxIndex], maxIndex };
}

NeuralNetwork::~NeuralNetwork()
{
	for (const LayerOptions& layer : m_LayerOptions)
		delete layer.activationFunction;
}

std::vector<double> NeuralNetwork::FeedForward(const std::vector<double>& input) const
{
	Matrix inputMatrix(input);
	for (unsigned int i = 0; i < m_WeightMatrices.size(); ++i)
	{
		Matrix outputMatrix = m_WeightMatrices[i] * inputMatrix;
		outputMatrix += m_Biases[i];
		if (m_LayerOptions[i].activationFunction != nullptr)
			outputMatrix.MapFunction(m_LayerOptions[i].activationFunction);
		inputMatrix = outputMatrix;
	}
	return inputMatrix.GetColumnVector();
}

void NeuralNetwork::GradientDescent(const int epochs, double learningRate, const int batchSize)
{
	for (int i = 1; i <= epochs; i++)
	{
		// std::cout << "Iteration: " << i << "Cost: " << std::endl; //costFunction()

		for (unsigned int i = 0; i < m_WeightMatrices.size(); i++) // za batchSize = 0
		{
			// zero gradients
			// get output
			// calculate loss
			// backProp()
			// ... m_WeightMatrices[i] = m_WeightMatrices[i] - learningRate * layerGradients
		}
	}
}

