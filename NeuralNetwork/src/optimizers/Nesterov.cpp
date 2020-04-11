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

#include "Optimizers.h"

namespace nn
{
	namespace optimizer
	{
		Nesterov::Nesterov(double lr, double momentum) : Optimizer(lr), m_Momentum(momentum)
		{

		}

		void Nesterov::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			Matrix previousWeight;
			Matrix previousBias;
			if (lastMomentWeight.find(layerIndex) == lastMomentWeight.end())
			{
				previousWeight = Matrix::Map(deltaWeight, [](double x) { return 0; });
				previousBias = Matrix::Map(deltaBias, [](double x) { return 0; });
				lastMomentWeight[layerIndex] = -m_LearningRate*deltaWeight;
				lastMomentBias[layerIndex] = -m_LearningRate*deltaBias;
			}
			else
			{
				previousWeight = lastMomentWeight[layerIndex];
				previousBias = lastMomentBias[layerIndex];
				lastMomentWeight[layerIndex] = m_Momentum * lastMomentWeight[layerIndex] - m_LearningRate * deltaWeight;
				lastMomentBias[layerIndex] = m_Momentum * lastMomentBias[layerIndex] - m_LearningRate * deltaBias;
			}
			layer.WeightMatrix += -m_Momentum*previousWeight + (1 + m_Momentum)*lastMomentWeight[layerIndex];
			layer.BiasMatrix += -m_Momentum*previousBias + (1 + m_Momentum)*lastMomentBias[layerIndex];
		}

		void Nesterov::Reset()
		{
			lastMomentWeight.clear();
			lastMomentBias.clear();
		}
	}
}