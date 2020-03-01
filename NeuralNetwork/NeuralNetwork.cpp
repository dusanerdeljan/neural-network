#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <algorithm>
#include <unordered_map>


NeuralNetwork::NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers, Initialization::Initializer* initializer) 
	: m_InputSize(inputSize), m_Layers(layers), m_WeightInitializer(initializer)
{
	if (m_WeightInitializer != nullptr)
		for (Layer& layer : m_Layers)
			initializer->Initialize(layer.m_WeightMatrix);
}

NeuralNetwork::NeuralNetwork(NeuralNetwork && net)
	: m_InputSize(net.m_InputSize), m_Layers(std::move(net.m_Layers)), m_WeightInitializer(m_WeightInitializer)
{
	net.m_WeightInitializer = nullptr;
}

NeuralNetwork::Output NeuralNetwork::Eval(const std::vector<double>& input)
{
	std::vector<double> outputResults = FeedForward(input).GetColumnVector();
	unsigned int maxIndex = std::max_element(outputResults.begin(), outputResults.end()) - outputResults.begin();
	return{ outputResults[maxIndex], maxIndex };
}

void NeuralNetwork::SaveModel(const char * fileName) const
{
	std::ofstream outfile;
	outfile.open(fileName, std::ios::binary | std::ios::out);
	outfile.write((char*)&m_InputSize, sizeof(m_InputSize));
	unsigned int numLayer = m_Layers.size();
	outfile.write((char*)&numLayer, sizeof(numLayer));
	for (const Layer& layer : m_Layers)
		layer.SaveLayer(outfile);
	outfile.close();
}

NeuralNetwork NeuralNetwork::LoadModel(const char * fileName)
{
	std::ifstream infile;
	infile.open(fileName, std::ios::in | std::ios::binary);
	unsigned int inputSize;
	infile.read((char*)&inputSize, sizeof(inputSize));
	unsigned int layerCount;
	infile.read((char*)&layerCount, sizeof(layerCount));
	std::vector<Layer> layers;
	for (unsigned int i = 0; i < layerCount; ++i)
		layers.push_back(Layer::LoadLayer(infile));
	infile.close();
	return NeuralNetwork(inputSize, layers, nullptr);
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
		Adam(epochs, learningRate, trainingData, batchSize);
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
// ------------------------------------------------------------------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------------------------------------------------------------------
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
// ------------------------------------------------------------------------------------------------------------------------------------------------------


// ------------------------------------------------------------------------------------------------------------------------------------------------------
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
	for (int i = 1; i <= epochs; i++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		std::unordered_map<unsigned int, Matrix> gradSquaredW;
		std::unordered_map<unsigned int, Matrix> gradSquaredB;

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

				if (gradSquaredW.find(i) == gradSquaredW.end())
				{
					gradSquaredW[i] = Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[i] = Matrix::Map(deltaBias, [](double x) { return x*x; });
				}
				else
				{
					gradSquaredW[i] = gradSquaredW[i] + Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[i] = gradSquaredB[i] + Matrix::Map(deltaBias, [](double x) { return x*x; });
				}

				Matrix deljenikW = Matrix::Map(gradSquaredW[i], [](double x) { return sqrt(x); }) + Matrix(gradSquaredW[i].GetHeight(), gradSquaredW[i].GetWidth(), 1e-7);
				Matrix deljenikB = Matrix::Map(gradSquaredB[i], [](double x) { return sqrt(x); }) + Matrix(gradSquaredB[i].GetHeight(), gradSquaredB[i].GetWidth(), 1e-7);

				m_Layers[i].m_WeightMatrix -= (learningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
				m_Layers[i].m_BiasMatrix -= (learningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));;
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
	double beta = 0.99;

	for (int i = 1; i <= epochs; i++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		std::unordered_map<unsigned int, Matrix> gradSquaredW;
		std::unordered_map<unsigned int, Matrix> gradSquaredB;

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

				if (gradSquaredW.find(i) == gradSquaredW.end())
				{
					gradSquaredW[i] = (1 - beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[i] = (1 - beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				}
				else
				{
					gradSquaredW[i] = beta * gradSquaredW[i] + (1 - beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[i] = beta * gradSquaredB[i] + (1 - beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				}
				
				Matrix deljenikW = Matrix::Map(gradSquaredW[i], [](double x) { return sqrt(x); }) + Matrix(gradSquaredW[i].GetHeight(), gradSquaredW[i].GetWidth(), 1e-7);
				Matrix deljenikB = Matrix::Map(gradSquaredB[i], [](double x) { return sqrt(x); }) + Matrix(gradSquaredB[i].GetHeight(), gradSquaredB[i].GetWidth(), 1e-7);

				m_Layers[i].m_WeightMatrix -= (learningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
				m_Layers[i].m_BiasMatrix -= (learningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));
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
	double beta = 0.9;
	double epsilon = 1e-7;

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
				for (unsigned int i = 0; i < prev.GetWidth(); i++)
					for (unsigned int j = 0; j < prev.GetHeight(); j++)
						s1 += prev(j, i);

				double s2 = 0;
				for (unsigned int i = 0; i < current.GetWidth(); i++)
					for (unsigned int j = 0; j < current.GetHeight(); j++)
						s2 += current(j, i);

				double delta = pow(s1 / ((prev.GetWidth() * prev.GetHeight())) - s2 / (current.GetWidth() * current.GetHeight()), 2);

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
	double beta1 = 0.9;
	double beta2 = 0.999;

	for (int epoch = 1; epoch <= epochs; epoch++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;

		// Weights
		std::unordered_map<unsigned int, Matrix> firstMomentW;
		std::unordered_map<unsigned int, Matrix> secondMomentW;

		// Biases
		std::unordered_map<unsigned int, Matrix> firstMomentB;
		std::unordered_map<unsigned int, Matrix> secondMomentB;

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

				Matrix firstUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());
				Matrix secondUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());

				Matrix firstUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());
				Matrix secondUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());

				if (firstMomentW.find(i) == firstMomentW.end())
				{
					// Weights
					firstMomentW[i] = (1 - beta1) * deltaWeight;
					secondMomentW[i] = (1 - beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					firstUnbiasW = firstMomentW[i] / (1 - pow(beta1, epoch));
					secondUnbiasW = secondMomentW[i] / (1 - pow(beta2, epoch));

					// Biases
					firstMomentB[i] = (1 - beta1) * deltaBias;
					secondMomentB[i] = (1 - beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					firstUnbiasB = firstMomentB[i] / (1 - pow(beta1, epoch));
					secondUnbiasB = secondMomentB[i] / (1 - pow(beta2, epoch));
				}
				else
				{
					// Weights
					firstMomentW[i] = firstMomentW[i] * beta1 + (1 - beta1) * deltaWeight;
					secondMomentW[i] = secondMomentW[i] * beta2 + (1 - beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					firstUnbiasW = firstMomentW[i] / (1 - pow(beta1, epoch));
					secondUnbiasW = secondMomentW[i] / (1 - pow(beta2, epoch));

					// Biases
					firstMomentB[i] = firstMomentB[i] * beta1 + (1 - beta1) * deltaBias;
					secondMomentB[i] = secondMomentB[i] * beta2 + (1 - beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					firstUnbiasB = firstMomentB[i] / (1 - pow(beta1, epoch));
					secondUnbiasB = secondMomentB[i] / (1 - pow(beta2, epoch));
				}

				Matrix deljenikW = Matrix::Map(secondUnbiasW, [](double x) { return sqrt(x); }) + Matrix(secondUnbiasW.GetHeight(), secondUnbiasW.GetWidth(), 1e-7);				
				Matrix weight = (learningRate * firstUnbiasW).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1/x; }));

				Matrix deljenikB = Matrix::Map(secondUnbiasB, [](double x) { return sqrt(x); }) + Matrix(secondUnbiasB.GetHeight(), secondUnbiasB.GetWidth(), 1e-7);
				Matrix bias = (learningRate * firstUnbiasB).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));

				m_Layers[i].m_WeightMatrix -= weight;
				m_Layers[i].m_BiasMatrix -= bias;
				error = Matrix::Transpose(m_Layers[i].m_WeightMatrix) * error;
			}
			numLoss++;
		}
		std::cout << "Epoch: " << epoch << " Loss: " << fullLoss / numLoss << std::endl;
	}
}
// -----------------------------------------------------------------------------------------------------------------------------------------------------


