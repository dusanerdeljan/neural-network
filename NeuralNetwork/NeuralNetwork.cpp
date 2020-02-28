#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <algorithm>
#include <functional>

NeuralNetwork::NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers) : m_InputSize(inputSize), m_Layers(layers)
{

}

NeuralNetwork::Output NeuralNetwork::Predict(const std::vector<double>& input)
{
	std::vector<double> outputResults = FeedForward(input).GetColumnVector();
	unsigned int maxIndex = std::max_element(outputResults.begin(), outputResults.end()) - outputResults.begin();
	return{ outputResults[maxIndex], maxIndex };
}

NeuralNetwork::~NeuralNetwork()
{
	
}

Matrix NeuralNetwork::FeedForward(const std::vector<double>& input)
{
	Matrix inputMatrix(input);
	for (Layer& layer : m_Layers)
	{
		inputMatrix = layer.UpdateActivation(inputMatrix);
	}
	return inputMatrix;
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

void NeuralNetwork::SGD(const int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData)
{
	for (int i = 1; i <= epochs; i++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		std::vector<NeuralNetwork::TrainingData> temp(trainingData);
		std::random_shuffle(temp.begin(), temp.end());

		for (auto trainIterator = temp.begin(); trainIterator != temp.end(); ++trainIterator)
		{
			Matrix prediction = FeedForward(trainIterator->inputs);
			Matrix error = prediction;
			error -= trainIterator->target;
			Matrix loss = Matrix::Map(error, [](double x) { return x*x; });
			fullLoss += loss(0, 0);
			for (int i = m_Layers.size() - 1; i >= 0; --i)
			{
				Matrix gradient(m_Layers[i].m_PreActivation);
				gradient.MapDerivative(m_Layers[i].m_ActivationFunction);
				gradient.DotProduct(error);
				gradient *= 2 * learningRate;
				Matrix previousActivation = i == 0 ? Matrix(trainIterator->inputs) : m_Layers[i - 1].m_Activation;
				m_Layers[i].m_WeightMatrix -= gradient * previousActivation.Transpose();
				m_Layers[i].m_BiasMatrix -= gradient;
				error = Matrix::Transpose(m_Layers[i].m_WeightMatrix) * error;
			}
			numLoss++;
		}
		std::cout << "Epoch: " << i << " Loss: " << fullLoss / numLoss << std::endl;
	}
}

