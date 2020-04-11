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
#include <unordered_map>
#include "../layers/Layer.h"

namespace nn
{
	namespace optimizer
	{
		enum class Type
		{
			GRADIENT_DESCENT, MOMENTUM, NESTEROV, ADAGRAD, RMSPROP, ADADELTA, ADAM, NADAM, ADAMAX, AMSGRAD
		};

		class Optimizer
		{
		protected:
			double m_LearningRate;
			Optimizer(double lr) : m_LearningRate(lr) {}
		public:
			virtual void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) = 0;
			virtual void Reset() {}
		};

		class GradientDescent : public Optimizer
		{
		public:
			GradientDescent(double lr);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
		};

		class Momentum : public Optimizer
		{
		private:
			double m_Momentum;
			std::unordered_map<unsigned int, Matrix> lastDeltaWeight;
			std::unordered_map<unsigned int, Matrix> lastDeltaBias;
		public:
			Momentum(double lr, double momentum = 0.9);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class Nesterov : public Optimizer
		{
		private:
			double m_Momentum;
			std::unordered_map<unsigned int, Matrix> lastMomentWeight;
			std::unordered_map<unsigned int, Matrix> lastMomentBias;
		public:
			Nesterov(double lr, double momentum = 0.9);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class Adagrad : public Optimizer
		{
		private:
			std::unordered_map<unsigned int, Matrix> gradSquaredW;
			std::unordered_map<unsigned int, Matrix> gradSquaredB;
		public:
			Adagrad(double lr);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class RMSProp : public Optimizer
		{
		private:
			double m_Beta;
			std::unordered_map<unsigned int, Matrix> gradSquaredW;
			std::unordered_map<unsigned int, Matrix> gradSquaredB;
		public:
			RMSProp(double lr, double beta = 0.99);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class Adadelta : public Optimizer
		{
		private:
			double m_Beta;
			std::unordered_map<unsigned int, Matrix> gradSquaredW;
			std::unordered_map<unsigned int, Matrix> gradSquaredB;
		public:
			Adadelta(double lr, double beta = 0.99);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class Adam : public Optimizer
		{
		private:
			double m_Beta1;
			double m_Beta2;
			// Weights
			std::unordered_map<unsigned int, Matrix> firstMomentW;
			std::unordered_map<unsigned int, Matrix> secondMomentW;
			// Biases
			std::unordered_map<unsigned int, Matrix> firstMomentB;
			std::unordered_map<unsigned int, Matrix> secondMomentB;
		public:
			Adam(double lr, double beta1 = 0.9, double beta2 = 0.999);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class Nadam : public Optimizer
		{
		private:
			double m_Beta1;
			double m_Beta2;
			// Weights
			std::unordered_map<unsigned int, Matrix> firstMomentW;
			std::unordered_map<unsigned int, Matrix> secondMomentW;
			// Biases
			std::unordered_map<unsigned int, Matrix> firstMomentB;
			std::unordered_map<unsigned int, Matrix> secondMomentB;
		public:
			Nadam(double lr, double beta1 = 0.9, double beta2 = 0.999);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class Adamax : public Optimizer
		{
		private:
			double m_Beta1;
			double m_Beta2;
			// Weights
			std::unordered_map<unsigned int, Matrix> firstMomentW;
			std::unordered_map<unsigned int, Matrix> infinityNormW;
			// Biases
			std::unordered_map<unsigned int, Matrix> firstMomentB;
			std::unordered_map<unsigned int, Matrix> infinityNormB;
		public:
			Adamax(double lr, double beta1 = 0.9, double beta2 = 0.999);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};

		class AMSGrad : public Optimizer
		{
		private:
			double m_Beta1;
			double m_Beta2;
			// Weights
			std::unordered_map<unsigned int, Matrix> firstMomentW;
			std::unordered_map<unsigned int, Matrix> secondMomentW;
			std::unordered_map<unsigned int, Matrix> infinityNormW;
			// Biases
			std::unordered_map<unsigned int, Matrix> firstMomentB;
			std::unordered_map<unsigned int, Matrix> secondMomentB;
			std::unordered_map<unsigned int, Matrix> infinityNormB;
		public:
			AMSGrad(double lr, double beta1 = 0.9, double beta2 = 0.999);
			void UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex = 0, unsigned int epoch = 0) override;
			void Reset() override;
		};
	}
}