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

	std::unordered_map<unsigned int, std::pair<Matrix, Matrix>> NeuralNetwork::Backpropagation(const std::vector<TrainingData>& batch, double & loss, unsigned int& numLoss)
	{
		std::unordered_map<unsigned int, std::pair<Matrix, Matrix>> deltaWeightBias;
		std::for_each(batch.begin(), batch.end(), [this, &loss, &numLoss, &deltaWeightBias](const TrainingData& data)
		{
			Matrix prediction = FeedForward(data.inputs);
			Matrix error = m_LossFunction->GetDerivative(prediction, data.target);
			loss += m_LossFunction->GetLoss(prediction, data.target);
			unsigned int layerIndex = m_Layers.size() - 1;
			std::for_each(m_Layers.rbegin(), m_Layers.rend(), [this, &error, &layerIndex, &data, &deltaWeightBias](Layer& layer)
			{
				Matrix gradient = m_LossFunction->Backward(layer, error);
				Matrix previousActivation = GetPreviousActivation(layerIndex, data.inputs);
				if (deltaWeightBias.find(layerIndex) == deltaWeightBias.end())
				{
					deltaWeightBias[layerIndex] = std::make_pair(gradient*previousActivation.Transpose(), gradient);
				}
				else
				{
					deltaWeightBias[layerIndex].first += gradient*previousActivation.Transpose();
					deltaWeightBias[layerIndex].second += gradient;
				}
				m_LossFunction->PropagateError(layer, error);
				layerIndex--;
			});
			numLoss++;
		});
		std::for_each(deltaWeightBias.begin(), deltaWeightBias.end(), [len = batch.size()]
		(std::pair<const unsigned int, std::pair<Matrix, Matrix>>& dWB) { dWB.second.first /= len; dWB.second.second /= len; });
		return deltaWeightBias;
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
			std::vector<TrainingData>::iterator batchEnd;
			for (std::vector<TrainingData>::iterator batchBegin = temp.begin(); batchBegin != temp.end(); batchBegin = batchEnd)
			{
				batchEnd = (unsigned)(temp.end() - batchBegin) >= batchSize ? batchBegin + batchSize : temp.end();
				std::vector<TrainingData> batch{ batchBegin, batchEnd };
				std::unordered_map<unsigned int, std::pair<Matrix, Matrix>> deltaWeightBias = Backpropagation(batch, fullLoss, numLoss);
				int layerIndex = m_Layers.size() - 1;
				std::for_each(m_Layers.rbegin(), m_Layers.rend(), [this, &batch, &deltaWeightBias, epoch, &optimizer, &layerIndex](Layer& layer)
				{
					optimizer.UpdateLayer(layer, deltaWeightBias[layerIndex].first, deltaWeightBias[layerIndex].second, layerIndex--, epoch);
				});
			}
			std::cout << "Epoch: " << epoch << " Loss: " << fullLoss / numLoss << std::endl;
		}
	}
}