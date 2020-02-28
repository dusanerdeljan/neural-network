#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <algorithm>
#include <functional>


NeuralNetwork::NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers) : m_InputSize(inputSize), m_Layers(layers)
{

}

NeuralNetwork::Output NeuralNetwork::Eval(const std::vector<double>& input)
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

void NeuralNetwork::Train(Optimizer::Type optimizer, unsigned int epochs,  double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	switch (optimizer)
	{
	case Optimizer::Type::SGD:
		SGD(epochs, learningRate, trainingData, batchSize);
		break;
	case Optimizer::Type::ADAGRAD:
		Adagrad(epochs, learningRate, trainingData, batchSize);
		break;
	case Optimizer::Type::ADAM:
		std::cout << "Need to be implemented" << std::endl;
		break;
	default:
		std::cout << "Default" << std::endl;
		break;
	}
}

// -----------------------------------------------------------------------------------------------------------------------------------------------------
// Optimizers
// -----------------------------------------------------------------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------------------------------------------------------------------------
void NeuralNetwork::SGD(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
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
// -----------------------------------------------------------------------------------------------------------------------------------------------------


// -----------------------------------------------------------------------------------------------------------------------------------------------------
void NeuralNetwork::Adam(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{

	std::cout << "Adam need to be implemented" << std::endl;

}
// -----------------------------------------------------------------------------------------------------------------------------------------------------


// -----------------------------------------------------------------------------------------------------------------------------------------------------
void NeuralNetwork::Adagrad(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	double epsilon = 0.01;
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

				Matrix alpha = Matrix::Map(gradient, [](double x) { return x*x; });
				alpha += Matrix(alpha.GetHeight(), alpha.GetWidth(), epsilon);
				Matrix deljenik = Matrix::Map(alpha, [](double x) {return sqrt(x); });
				Matrix newLearningRate = Matrix::Map(deljenik, [learningRate](double x) { return learningRate/x; });
				Matrix newGradient(gradient.GetHeight(), 1, 0);
				for (unsigned int i = 0; i < gradient.GetColumnVector().size(); i++)
					newGradient(i, 0) = gradient(i, 0) * newLearningRate(i, 0) * 2;

				Matrix previousActivation = i == 0 ? Matrix(trainIterator->inputs) : m_Layers[i - 1].m_Activation;
				m_Layers[i].m_WeightMatrix -= newGradient * previousActivation.Transpose();
				m_Layers[i].m_BiasMatrix -= newGradient;
				error = Matrix::Transpose(m_Layers[i].m_WeightMatrix) * error;
			}
			numLoss++;
		}
		std::cout << "Epoch: " << i << " Loss: " << fullLoss / numLoss << std::endl;
	}
}
// -----------------------------------------------------------------------------------------------------------------------------------------------------


