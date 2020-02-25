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

// Trebamo napraviti interfejs i za ovo, loss functions tako nesto
// Ovo input i labels se moze kasnije objediniti u jednu strukturu
Matrix NeuralNetwork::MeanAbsoluteError(const std::vector<double>& input, double target)
{
	//float total = 0;
	//for (int i = 0; i < input.size(); ++i)
	//{
	//	float estimated_y = 100; // nn_output hardcoded, can't use predict
	//	float absolute_error = abs(estimated_y - input[i]);
	//	total += absolute_error;
	//}
	//float mean_absolute_error = total / input.size();

	//return mean_absolute_error;
	Matrix estimatedMatrix = FeedForward(input);
	Matrix labelMatrix = Matrix::BuildColumnMatrix(estimatedMatrix.GetHeight(), target);
	return Matrix::Map((estimatedMatrix - labelMatrix), [](double x)
	{
		return abs(x);
	}) / input.size();
}

Matrix NeuralNetwork::MeanSquaredError(const std::vector<double>& input, double target)
{
	//float total = 0;
	//for (int i = 0; i < input.size(); ++i)
	//{
	//	float estimated_y = 100; // nn_output hardcoded, can't use predict
	//	float squared_error = pow(estimated_y - input[i], 2);
	//	total += squared_error;
	//}
	//float mean_squared_error = total / input.size();

	//return mean_squared_error;
	Matrix estimatedMatrix = FeedForward(input);
	Matrix labelMatrix = Matrix::BuildColumnMatrix(estimatedMatrix.GetHeight(), target);
	return Matrix::Map((estimatedMatrix - labelMatrix), [](double x)
	{
		return pow(x, 2);
	}) / input.size();
}


// Moramo se dogovoriti kako cemo layerGradients implementirati
void NeuralNetwork::SGD(const int epochs, double learningRate, const std::vector<std::vector<double>>& inputs, const std::vector<double>& labels)
{
	float val = 0;
	for (int i = 1; i <= epochs; i++)
	{
		auto labelIterator = labels.begin();
		for (auto inputsIterator = inputs.begin(); inputsIterator != inputs.end(); ++inputsIterator, ++labelIterator)
		{
			Matrix loss = MeanAbsoluteError(*inputsIterator, *labelIterator);
#ifdef _DEBUG
			//std::cout << "Epoch: " << i << " Loss: " << loss;
#endif // _DEBUG
			std::vector<Matrix> layerOutputs = TrainFeedForward(*inputsIterator); // ovo se bespotrebno 2 puta racuna, ali neka ga za sada
			for (unsigned int i = m_WeightMatrices.size()-1; i >= 1; --i)
			{
				Matrix layerOutput = layerOutputs[i];
				Matrix layerGradient = layerOutput.MapDerivative(m_LayerOptions[i].activationFunction) * loss * learningRate;
				Matrix deltaWeight =  layerGradient * Matrix::Transpose(layerOutputs[i - 1]);
				m_WeightMatrices[i] += deltaWeight;
				m_Biases[i] += layerGradient;
				loss = Matrix::Transpose(m_WeightMatrices[i]) * loss;
			}
		}
	}
}

