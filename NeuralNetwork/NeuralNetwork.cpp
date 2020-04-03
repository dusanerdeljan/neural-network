#include "NeuralNetwork.h"
#include "ActivationFunctions.h"
#include <algorithm>
#include <unordered_map>

namespace nn
{
	NeuralNetwork::NeuralNetwork(unsigned int inputSize, std::vector<Layer>&& layers, initialization::Type initializer, loss::Type lossFunction)
		: m_InputSize(inputSize), m_Layers(layers), m_WeightInitializer(WeightInitializerFactory::BuildWeightInitializer(initializer)), m_LossFunction(LossFunctionFactory::BuildLossFunction(lossFunction))
	{
		if (m_WeightInitializer != nullptr)
			std::for_each(m_Layers.begin(), m_Layers.end(), [wi = m_WeightInitializer](Layer& layer) { layer.Initialize(wi); });
	}

	NeuralNetwork::NeuralNetwork(NeuralNetwork && net)
		: m_InputSize(net.m_InputSize), m_Layers(std::move(net.m_Layers)), m_WeightInitializer(m_WeightInitializer), m_LossFunction(net.m_LossFunction)
	{
		net.m_WeightInitializer = nullptr;
	}

	Output NeuralNetwork::Eval(const std::vector<double>& input)
	{
		std::vector<double> outputResults = FeedForward(input).GetColumnVector();
		unsigned int maxIndex = std::max_element(outputResults.begin(), outputResults.end()) - outputResults.begin();
		return{ outputResults[maxIndex], maxIndex };
	}

	Output NeuralNetwork::operator()(const std::vector<double>& input)
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
		loss::Type type = m_LossFunction->GetType();
		outfile.write((char*)&type, sizeof(loss::Type));
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
		return NeuralNetwork(inputSize, std::move(layers), initialization::NONE, loss::Type(lossType));
	}

	Matrix NeuralNetwork::FeedForward(const std::vector<double>& input)
	{
		Matrix inputMatrix(input);
		std::for_each(m_Layers.begin(), m_Layers.end(), [&inputMatrix](Layer& layer) { inputMatrix = layer.UpdateActivation(inputMatrix); });
		return inputMatrix;
	}

	inline Matrix NeuralNetwork::GetPreviousActivation(int layerIndex, const std::vector<double>& inputs) const
	{
		return layerIndex == 0 ? Matrix(inputs) : m_Layers[layerIndex - 1].m_Activation;
	}

	void NeuralNetwork::Train(optimizer::Optimizer& optimizer, unsigned int epochs, const std::vector<TrainingData>& trainingData, unsigned int batchSize)
	{
		for (unsigned int epoch = 1; epoch <= epochs; epoch++)
		{
			double fullLoss = 0;
			unsigned int numLoss = 0;
			optimizer.Reset();
			std::vector<TrainingData> temp(trainingData);
			std::random_shuffle(temp.begin(), temp.end());
			std::for_each(temp.begin(), temp.end(), [this, &fullLoss, &optimizer, epoch, &numLoss](const TrainingData& data)
			{
				Matrix prediction = FeedForward(data.inputs);
				Matrix error = m_LossFunction->GetDerivative(prediction, data.target);
				fullLoss += m_LossFunction->GetLoss(prediction, data.target);
				int layerIndex = m_Layers.size() - 1;
				std::for_each(m_Layers.rbegin(), m_Layers.rend(), [this, &data, &error, epoch, &optimizer, &layerIndex](Layer& layer)
				{
					Matrix gradient = m_LossFunction->Backward(layer, error);
					Matrix previousActivation = GetPreviousActivation(layerIndex, data.inputs);
					optimizer.UpdateLayer(layer, gradient, previousActivation, layerIndex--, epoch);
					m_LossFunction->PropagateError(layer, error);
				});
				numLoss++;
			});
			std::cout << "Epoch: " << epoch << " Loss: " << fullLoss / numLoss << std::endl;
		}
	}
}