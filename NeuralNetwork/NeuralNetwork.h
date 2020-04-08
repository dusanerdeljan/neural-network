#pragma once
#include <vector>
#include <unordered_map>
#include "src/layers/Layer.h"
#include "src/optimizers/Optimizers.h"
#include "src/initializers/WeightInitializers.h"
#include "src/losses/LossFunctions.h"
#include "src/regularizers/Regularizers.h"

#ifdef _WINDLL // .dll or .lib
#define PYTHON_API
#endif // _WINDLL

namespace nn
{
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
		TrainingData& operator=(const TrainingData& data)
		{
			inputs = data.inputs;
			target = data.target;
			return *this;
		}
	};

	class NeuralNetwork
	{
	private:
		unsigned int m_InputSize;
		std::vector<Layer> m_Layers;
		std::shared_ptr<initialization::Initializer> m_WeightInitializer;
		std::shared_ptr<loss::LossFunction> m_LossFunction;

	public:
		NeuralNetwork(unsigned int inputSize, std::vector<Layer>&& layers, initialization::Type initializer, loss::Type lossFunction);
		NeuralNetwork(NeuralNetwork&& net);
		void Train(optimizer::Optimizer& optimizer, unsigned int epochs, const std::vector<TrainingData>& trainingData, unsigned int batchSize = 1, regularizer::Type regularizerType = regularizer::NONE);
		Output Eval(const std::vector<double>& input);
		Output operator()(const std::vector<double>& input);
		void SaveModel(const char* fileName) const;
		static NeuralNetwork LoadModel(const char* fileName);
	private:
		Matrix FeedForward(const std::vector<double>& input);
		inline Matrix GetPreviousActivation(int layerIndex, const std::vector<double>& data) const;
		std::unordered_map<unsigned int, std::pair<Matrix, Matrix>> Backpropagation(const std::vector<TrainingData>& batch, double& loss, unsigned int& numLoss);
	};
}

#ifdef PYTHON_API
#include "python/PythonAPI.h"
#endif // PYTHON_API
