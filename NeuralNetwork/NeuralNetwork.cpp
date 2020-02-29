#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <algorithm>
#include <unordered_map>


NeuralNetwork::NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers, Initialization::Initializer* initializer) 
	: m_InputSize(inputSize), m_Layers(layers), m_WeightInitializer(initializer)
{
	for (Layer& layer : m_Layers)
		initializer->Initialize(layer.m_WeightMatrix);
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
	case Optimizer::Type::RMSPROP:
		RMSprop(epochs, learningRate, trainingData, batchSize);
		break;
	case Optimizer::Type::ADADELTA:
		Adadelta(epochs, trainingData, batchSize);
		break;
	case Optimizer::Type::ADAM:
		std::cout << "Need to be implemented" << std::endl;
		break;
	case Optimizer::Type::MOMENTUM:
		Momentum(epochs, learningRate, trainingData, batchSize);
		break;
	case Optimizer::Type::NESTEROV:
		Nesterov(epochs, learningRate, trainingData, batchSize);
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

void NeuralNetwork::Momentum(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	double momentum = 0.9;
	for (int i = 1; i <= epochs; i++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		std::unordered_map<unsigned int, Matrix> lastDeltaWeight;
		std::unordered_map<unsigned int, Matrix> lastDeltaBias;

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
				Matrix previousActivation = i == 0 ? Matrix(trainIterator->inputs) : m_Layers[i - 1].m_Activation;

				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;
				if (lastDeltaWeight.find(i) == lastDeltaWeight.end())
				{
					lastDeltaWeight[i] = deltaWeight*learningRate;
					lastDeltaBias[i] = deltaBias*learningRate;
				}
				else
				{
					lastDeltaWeight[i] *= momentum;
					lastDeltaWeight[i] += deltaWeight * learningRate;
					lastDeltaBias[i] *= momentum;
					lastDeltaBias[i] += deltaBias * learningRate;
				}

				m_Layers[i].m_WeightMatrix -= lastDeltaWeight[i];
				m_Layers[i].m_BiasMatrix -= lastDeltaBias[i];
				error = Matrix::Transpose(m_Layers[i].m_WeightMatrix) * error;
			}
			numLoss++;
		}
		std::cout << "Epoch: " << i << " Loss: " << fullLoss / numLoss << std::endl;
	}
}

void NeuralNetwork::Nesterov(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	double momentum = 0.9;
	for (int i = 1; i <= epochs; i++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		std::unordered_map<unsigned int, Matrix> lastDeltaWeight;
		std::unordered_map<unsigned int, Matrix> lastDeltaBias;

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
				Matrix previousActivation = i == 0 ? Matrix(trainIterator->inputs) : m_Layers[i - 1].m_Activation;

				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;
				if (lastDeltaWeight.find(i) == lastDeltaWeight.end())
				{
					lastDeltaWeight[i] = Matrix::Map(deltaWeight, [](double x) { return 0.0; });
					lastDeltaBias[i] = Matrix::Map(deltaBias, [](double x) { return 0.0; });
				}
				Matrix tempWeight = lastDeltaWeight[i];
				lastDeltaWeight[i] *= momentum;
				lastDeltaWeight[i] -= deltaWeight * learningRate;
				Matrix tempBias = lastDeltaBias[i];
				lastDeltaBias[i] *= momentum;
				lastDeltaBias[i] -= deltaBias*learningRate;
				m_Layers[i].m_WeightMatrix += (tempWeight*(-momentum)) + (lastDeltaWeight[i] * (1 + momentum));
				m_Layers[i].m_BiasMatrix += (tempBias*(-momentum)) + (lastDeltaBias[i] * (1 + momentum));
				error = Matrix::Transpose(m_Layers[i].m_WeightMatrix) * error;
			}
			numLoss++;
		}
		std::cout << "Epoch: " << i << " Loss: " << fullLoss / numLoss << std::endl;
	}
}
// -----------------------------------------------------------------------------------------------------------------------------------------------------


// -----------------------------------------------------------------------------------------------------------------------------------------------------
void NeuralNetwork::Adagrad(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	double epsilon = 10e-4;
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
				Matrix newLearningRate = Matrix::Map(deljenik, [learningRate](double x) { return learningRate / x; });
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


// -----------------------------------------------------------------------------------------------------------------------------------------------------
void NeuralNetwork::RMSprop(unsigned int epochs, double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	int beta = 0.95;
	int epsilon = 10e-6;

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
			double s = 0;

			for (int i = m_Layers.size() - 1; i >= 0; --i)
			{
				Matrix gradient(m_Layers[i].m_PreActivation);
				gradient.MapDerivative(m_Layers[i].m_ActivationFunction);

				Matrix temp(Matrix::Map(gradient, [](double x) { return x*x; }));

				double sum = 0;
				for (unsigned int i = 0; i < temp.GetHeight(); i++)
					sum += temp(i, 0);

				s = beta * s + (1-beta) * (sum / temp.GetHeight());

				double update = learningRate / sqrt(s + epsilon);
				gradient *= 2 * update;

				gradient.DotProduct(error);
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
void NeuralNetwork::Adadelta(unsigned int epochs, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	double beta = 0.95;
	double epsilon = 10e-6;

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

			double d = 0;
			double s = 0;

			for (int i = m_Layers.size() - 1; i >= 0; --i)
			{
				Matrix gradient(m_Layers[i].m_PreActivation);
				gradient.MapDerivative(m_Layers[i].m_ActivationFunction);
				gradient.DotProduct(error);

				Matrix prev = i == 0 ? Matrix(trainIterator->inputs) : m_Layers[i - 1].m_Activation;
				prev = gradient * prev.Transpose();

				Matrix current = i == 0 ? Matrix(trainIterator->inputs) : m_Layers[i].m_Activation;
				current = gradient * current.Transpose();

				double s1 = 0;
				unsigned int num1 = 0;
				for (unsigned int i = 0; i < prev.GetWidth(); i++)
				{
					for (unsigned int j = 0; j < prev.GetHeight(); j++)
					{
						s1 += prev(j, i);
						num1++;
					}
				}

				double s2 = 0;
				unsigned int num2 = 0;
				for (unsigned int i = 0; i < current.GetWidth(); i++)
				{
					for (unsigned int j = 0; j < current.GetHeight(); j++)
					{
						s2 += current(j, i);
						num2++;
					}
				}

				double delta = pow(s1 / num1 - s2 / num2, 2);

				Matrix squared_grad(Matrix::Map(gradient, [](double x) { return x*x; }));
				double sumS = 0;
				for (unsigned int i = 0; i < squared_grad.GetHeight(); i++)
					sumS += squared_grad(i, 0);
				
				d = beta * d + (1 - beta) * delta;
				s = beta * s + (1 - beta) * (sumS / squared_grad.GetHeight());

				double update = sqrt(d + epsilon) / sqrt(s + epsilon);
				gradient *= 2 * update;

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


