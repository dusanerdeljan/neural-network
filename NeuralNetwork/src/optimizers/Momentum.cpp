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
		Momentum::Momentum(double lr, double momentum) : Optimizer(lr), m_Momentum(momentum)
		{

		}

		void Momentum::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			if (lastDeltaWeight.find(layerIndex) == lastDeltaWeight.end())
			{
				lastDeltaWeight[layerIndex] = (1 - m_Momentum) * deltaWeight;
				lastDeltaBias[layerIndex] = (1 - m_Momentum) * deltaBias;
			}
			else
			{
				lastDeltaWeight[layerIndex] = m_Momentum*lastDeltaWeight[layerIndex] + (1 - m_Momentum) * deltaWeight;
				lastDeltaBias[layerIndex] = m_Momentum*lastDeltaBias[layerIndex] + (1 - m_Momentum) * deltaBias;
			}
			layer.WeightMatrix -= m_LearningRate * lastDeltaWeight[layerIndex];
			layer.BiasMatrix -= m_LearningRate * lastDeltaBias[layerIndex];
		}

		void Momentum::Reset()
		{
			lastDeltaWeight.clear();
			lastDeltaBias.clear();
		}
	}
}