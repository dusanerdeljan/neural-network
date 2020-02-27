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
	inputMatrix.Transpose();
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

std::pair<std::vector<Matrix>, std::vector<Matrix>> NeuralNetwork::BackProp(const TrainingData & trainData) const
{
	std::vector<Matrix> weightProxies;
	std::vector<Matrix> biasProxies;
	InitializeProxies(weightProxies, biasProxies);

	return{ weightProxies, biasProxies };
}

void NeuralNetwork::InitializeProxies(std::vector<Matrix>& weightProxies, std::vector<Matrix>& biasProxies) const
{
	//std::for_each(m_WeightMatrices.begin(), m_WeightMatrices.end(), [&weightProxies](const Matrix& matrix)
	//{
	//	weightProxies.push_back(Matrix(matrix.GetHeight(), matrix.GetWidth(), 0));
	//});
	//std::for_each(m_Biases.begin(), m_Biases.end(), [&biasProxies](const Matrix& matrix)
	//{
	//	biasProxies.push_back(Matrix(matrix.GetHeight(), matrix.GetWidth(), 0));
	//});
}


void NeuralNetwork::SimpleTraining(const int epochs, const std::vector<NeuralNetwork::TrainingData>& trainingData)
{
	for (int i = 1; i <= epochs; i++)
	{
		for (auto trainIterator = trainingData.begin(); trainIterator != trainingData.end(); ++trainIterator)
		{
			ActivationFunctions::Sigmoid sigma;
			//std::cout << "Inputs: " << (*trainIterator).inputs[0] << " " << (*trainIterator).inputs[1] << std::endl;
			//std::cout << "Prediction: " << Predict((*trainIterator).inputs).value << std::endl;
			
			double target = (*trainIterator).target;

			double out = (Matrix::Transpose((*trainIterator).inputs) * m_Layers[0].m_WeightMatrix).GetColumnVector()[0];
			double output = sigma.Function(out);

			double error = pow((*trainIterator).target - output, 2);
			Matrix adjustments = (error * sigma.Derivative(out)) * (*trainIterator).inputs;
			m_Layers[0].m_WeightMatrix += adjustments;
		}
	}

	// Testing
	ActivationFunctions::Sigmoid sigma;
	double res0 = sigma.Function((Matrix::Transpose(std::vector<double>({ 1, 0 })) * m_Layers[0].m_WeightMatrix).GetColumnVector()[0]);
	double res1 = sigma.Function((Matrix::Transpose(std::vector<double>({ 1, 1 })) * m_Layers[0].m_WeightMatrix).GetColumnVector()[0]);
	double res2 = sigma.Function((Matrix::Transpose(std::vector<double>({ 0, 1 })) * m_Layers[0].m_WeightMatrix).GetColumnVector()[0]);
	double res3 = sigma.Function((Matrix::Transpose(std::vector<double>({ 0, 0 })) * m_Layers[0].m_WeightMatrix).GetColumnVector()[0]);
	std::cout << "1, 0 -> " << res0 << std::endl << "1, 1 -> " << res1 << std::endl << "0, 1 -> " << res2 << std::endl << "0, 0 -> " << res3 << std::endl;
}



void NeuralNetwork::SGD(const int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData)
{
	for (int i = 1; i <= epochs; i++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		std::vector<NeuralNetwork::TrainingData> temp(trainingData);
		std::random_shuffle(temp.begin(), temp.end());

		for (auto trainIterator = trainingData.begin(); trainIterator != trainingData.end(); ++trainIterator)
		{

			// Feed-forward
			Matrix prediction = FeedForward((*trainIterator).inputs);			
			std::cout << "Inputs: " << (*trainIterator).inputs[0] << " " << (*trainIterator).inputs[1] << std::endl;
			std::cout << "Prediction: " << prediction << std::endl;

			// Backpropagation
			double loss = pow((*trainIterator).target - prediction.GetColumnVector()[0], 2);
			double slope = 2 * ((*trainIterator).target - prediction.GetColumnVector()[0]);
			//double p = pred.GetColumnVector()[0] + learningRate * slope;
			
			for (auto& layer : m_Layers)
			{
				layer.m_WeightMatrix += Matrix::Transpose(Matrix((learningRate*slope) * Matrix(layer.m_WeightMatrix.GetWidth(), layer.m_WeightMatrix.GetHeight(), 1)));
				layer.m_WeightMatrix *= -1;
				layer.m_BiasMatrix += Matrix::Transpose(Matrix((learningRate*slope) * Matrix(layer.m_BiasMatrix.GetWidth(), layer.m_BiasMatrix.GetHeight(), 1)));
				layer.m_BiasMatrix *= -1;
			}

			fullLoss += loss;
			numLoss++;
		}
		std::cout << "Epoch: " << i << " Loss: " << fullLoss / numLoss << std::endl;
	}
}

