#pragma once
#include <vector>
#include "Layer.h"
#include "Optimizers.h"
#include "WeightInitializers.h"
#include "LossFunctions.h"

namespace nn
{
	class NeuralNetwork
	{
	public:
		struct Output
		{
			double value;
			unsigned int index;
			Output(double v, unsigned int i) : value(v), index(i) {}
		};

		struct TrainingData
		{
			std::vector<double> inputs;
			std::vector<double> target;
			TrainingData(const std::vector<double>& inputs, double target) : inputs(inputs), target({ target }) {}
			TrainingData(const std::vector<double>& inputs, const std::vector<double>& target) : inputs(inputs), target(target) {}
			TrainingData& operator=(const NeuralNetwork::TrainingData& data)
			{
				inputs = data.inputs;
				target = data.target;
				return *this;
			}
		};

	private:
		unsigned int m_InputSize;
		std::vector<Layer> m_Layers;
		std::shared_ptr<initialization::Initializer> m_WeightInitializer;
		std::shared_ptr<loss::LossFunction> m_LossFunction;

	public:
		NeuralNetwork(unsigned int inputSize, std::vector<Layer>&& layers, initialization::Type initializer, loss::Type lossFunction);
		NeuralNetwork(NeuralNetwork&& net);
		void Train(optimizer::Optimizer& optimizer, unsigned int epochs, const std::vector<NeuralNetwork::TrainingData>& trainingData, unsigned int batchSize = 1);
		Output Eval(const std::vector<double>& input);
		Output operator()(const std::vector<double>& input);
		void SaveModel(const char* fileName) const;
		static NeuralNetwork LoadModel(const char* fileName);
	private:
		Matrix FeedForward(const std::vector<double>& input);
		inline Matrix GetPreviousActivation(int layerIndex, const std::vector<double>& data) const;
	};
}

