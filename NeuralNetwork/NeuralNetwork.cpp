#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <algorithm>
#include <unordered_map>


NeuralNetwork::NeuralNetwork(unsigned int inputSize, const std::vector<Layer>& layers, Initialization::Initializer* initializer, Loss::LossFunction* lossFunction)
	: m_InputSize(inputSize), m_Layers(layers), m_WeightInitializer(initializer), m_LossFunction(lossFunction)
{
	if (m_WeightInitializer != nullptr)
		for (Layer& layer : m_Layers)
			initializer->Initialize(layer.m_WeightMatrix);
}

NeuralNetwork::NeuralNetwork(NeuralNetwork && net)
	: m_InputSize(net.m_InputSize), m_Layers(std::move(net.m_Layers)), m_WeightInitializer(m_WeightInitializer), m_LossFunction(net.m_LossFunction)
{
	net.m_WeightInitializer = nullptr;
}

NeuralNetwork::Output NeuralNetwork::Eval(const std::vector<double>& input)
{
	std::vector<double> outputResults = FeedForward(input).GetColumnVector();
	unsigned int maxIndex = std::max_element(outputResults.begin(), outputResults.end()) - outputResults.begin();
	return{ outputResults[maxIndex], maxIndex };
}

NeuralNetwork::Output NeuralNetwork::operator()(const std::vector<double>& input)
{
	return Eval(input);
}

void NeuralNetwork::SaveModel(const char * fileName) const
{
	std::ofstream outfile;
	outfile.open(fileName, std::ios::binary | std::ios::out);
	outfile.write((char*)&m_InputSize, sizeof(m_InputSize));
	unsigned int numLayer = m_Layers.size();
	outfile.write((char*)&numLayer, sizeof(numLayer));
	Loss::Type type = m_LossFunction->GetType();
	outfile.write((char*)&type, sizeof(Loss::Type));
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
	int lossType;
	infile.read((char*)&lossType, sizeof(lossType));
	std::vector<Layer> layers;
	for (unsigned int i = 0; i < layerCount; ++i)
		layers.push_back(Layer::LoadLayer(infile));
	infile.close();
	return NeuralNetwork(inputSize, std::move(layers), nullptr, LossFunctionFactory::BuildLossFunction(Loss::Type(lossType)));
}

NeuralNetwork::~NeuralNetwork()
{
	delete m_WeightInitializer;
	delete m_LossFunction;
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

inline Matrix NeuralNetwork::GetPreviousActivation(int layerIndex, std::vector<NeuralNetwork::TrainingData>::iterator trainIterator) const
{
	return layerIndex == 0 ? Matrix(trainIterator->inputs) : m_Layers[layerIndex - 1].m_Activation;
}

void NeuralNetwork::Train(Optimizer::Optimizer* optimizer, unsigned int epochs,  double learningRate, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize)
{
	for (unsigned int epoch = 1; epoch <= epochs; epoch++)
	{
		double fullLoss = 0;
		unsigned int numLoss = 0;
		optimizer->Reset();
		std::vector<NeuralNetwork::TrainingData> temp(trainingData);
		std::random_shuffle(temp.begin(), temp.end());

		for (auto trainIterator = temp.begin(); trainIterator != temp.end(); ++trainIterator)
		{
			Matrix prediction = FeedForward(trainIterator->inputs);
			Matrix error = m_LossFunction->GetDerivative(prediction, trainIterator->target);
			fullLoss += m_LossFunction->GetLoss(prediction, trainIterator->target);
			for (int i = m_Layers.size() - 1; i >= 0; --i)
			{
				Matrix gradient = m_LossFunction->Backward(m_Layers[i], error);
				Matrix previousActivation = GetPreviousActivation(i, trainIterator);
				optimizer->UpdateLayer(m_Layers[i], gradient, previousActivation, i, epoch);
				m_LossFunction->PropagateError(m_Layers[i], error);
			}
			numLoss++;
		}
		std::cout << "Epoch: " << epoch << " Loss: " << fullLoss / numLoss << std::endl;
	}
}