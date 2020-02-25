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
	std::vector<double> outputResults = FeedForward(input).GetColumnVector();
	unsigned int maxIndex = std::max_element(outputResults.begin(), outputResults.end()) - outputResults.begin();
	return{ outputResults[maxIndex], maxIndex };
}

NeuralNetwork::~NeuralNetwork()
{
	for (const LayerOptions& layer : m_LayerOptions)
		delete layer.activationFunction;
}

Matrix NeuralNetwork::FeedForward(const std::vector<double>& input) const
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
	return inputMatrix;
}

std::vector<Matrix> NeuralNetwork::TrainFeedForward(const std::vector<double>& input) const
{
	std::vector<Matrix> layerOutputs;
	Matrix inputMatrix(input);
	for (unsigned int i = 0; i < m_WeightMatrices.size(); ++i)
	{
		Matrix outputMatrix = m_WeightMatrices[i] * inputMatrix;
		outputMatrix += m_Biases[i];
		if (m_LayerOptions[i].activationFunction != nullptr)
			outputMatrix.MapFunction(m_LayerOptions[i].activationFunction);
		inputMatrix = outputMatrix;
		layerOutputs.push_back(outputMatrix);
	}
	return layerOutputs;
}

Matrix NeuralNetwork::MeanAbsoluteError(const NeuralNetwork::TrainingData& trainData)
{
	Matrix estimatedMatrix = FeedForward(trainData.inputs);
	Matrix labelMatrix = Matrix::BuildColumnMatrix(estimatedMatrix.GetHeight(), trainData.target);
	return Matrix::Map((estimatedMatrix - labelMatrix), [](double x)
	{
		return abs(x);
	}) / trainData.inputs.size();
}

Matrix NeuralNetwork::MeanSquaredError(const NeuralNetwork::TrainingData& trainData)
{
	Matrix estimatedMatrix = FeedForward(trainData.inputs);
	Matrix labelMatrix = Matrix::BuildColumnMatrix(estimatedMatrix.GetHeight(), trainData.target);
	return Matrix::Map((estimatedMatrix - labelMatrix), [](double x)
	{
		return pow(x, 2);
	}) / trainData.inputs.size();
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> NeuralNetwork::BackProp(const TrainingData & trainData) const
{
	std::vector<Matrix> weightProxies;
	std::vector<Matrix> biasProxies;
	InitializeProxies(weightProxies, biasProxies);

	return{ weightProxies, biasProxies };
}

void NeuralNetwork::InitializeProxies(std::vector<Matrix>& weightProxies, std::vector<Matrix>& biasProxies) const
{
	std::for_each(m_WeightMatrices.begin(), m_WeightMatrices.end(), [&weightProxies](const Matrix& matrix)
	{
		weightProxies.push_back(Matrix(matrix.GetHeight(), matrix.GetWidth(), 0));
	});
	std::for_each(m_Biases.begin(), m_Biases.end(), [&biasProxies](const Matrix& matrix)
	{
		biasProxies.push_back(Matrix(matrix.GetHeight(), matrix.GetWidth(), 0));
	});
}


// Moramo se dogovoriti kako cemo layerGradients implementirati
void NeuralNetwork::SGD(const int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData)
{
	for (int i = 1; i <= epochs; i++)
	{
		// std::random_shuffle(trainingData.begin(), trainingData.end());
		std::vector<Matrix> weightProxies;
		std::vector<Matrix> biasProxies;
		InitializeProxies(weightProxies, biasProxies);
		// Ignore batch size for now, will be added later
		for (auto trainIterator = trainingData.begin(); trainIterator != trainingData.end(); ++trainIterator)
		{
			Matrix loss = MeanAbsoluteError(*trainIterator);
			std::vector<Matrix> layerOutputs = TrainFeedForward(trainIterator->inputs);
			for (unsigned int i = m_WeightMatrices.size() - 1; i >= 1; --i)
			{
				Matrix layerOutput = layerOutputs[i];
				Matrix layerGradient = layerOutput.MapDerivative(m_LayerOptions[i].activationFunction) * loss * learningRate;
				Matrix deltaWeight = layerGradient * Matrix::Transpose(layerOutputs[i - 1]);
				m_WeightMatrices[i] += deltaWeight;
				m_Biases[i] += layerGradient;
				loss = Matrix::Transpose(m_WeightMatrices[i]) * loss;
			}
		}
	}
}

