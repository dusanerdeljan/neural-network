/*
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

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
		double Value;
		unsigned int Argmax;
		Output(double v, unsigned int i) : Value(v), Argmax(i) {}
	};

	struct TrainingData
	{
		std::vector<double> Inputs;
		std::vector<double> Target;
		TrainingData(const std::vector<double>& inputs, double target) : Inputs(inputs), Target({ target }) {}
		TrainingData(const std::vector<double>& inputs, const std::vector<double>& target) : Inputs(inputs), Target(target) {}
		TrainingData& operator=(const TrainingData& data)
		{
			Inputs = data.Inputs;
			Target = data.Target;
			return *this;
		}
		TrainingData(const TrainingData& data) noexcept : Inputs(data.Inputs), Target(data.Target) {}
		TrainingData(TrainingData&& data) noexcept : Inputs(data.Inputs), Target(data.Target) {}
		TrainingData(std::vector<double>&& inputs, std::vector<double>&& target) noexcept : Inputs(inputs), Target(target) {}
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
		NeuralNetwork& operator=(NeuralNetwork&& net);
		NeuralNetwork(NeuralNetwork&& net);
		void Train(optimizer::Optimizer& optimizer, unsigned int epochs, const std::vector<TrainingData>& trainingData, unsigned int batchSize = 1, regularizer::Type regularizerType = regularizer::NONE);
		Output Eval(const std::vector<double>& input);
		Output Eval(std::vector<double>&& input);
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
