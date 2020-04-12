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
		Adadelta::Adadelta(double lr, double beta) : Optimizer(lr), m_Beta(beta)
		{

		}

		void Adadelta::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			if (gradSquaredW.find(layerIndex) == gradSquaredW.end())
			{
				gradSquaredW[layerIndex] = (1 - m_Beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				gradSquaredB[layerIndex] = (1 - m_Beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
			}
			else
			{
				gradSquaredW[layerIndex] = m_Beta * gradSquaredW[layerIndex] + (1 - m_Beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				gradSquaredB[layerIndex] = m_Beta * gradSquaredB[layerIndex] + (1 - m_Beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
			}
			layer.WeightMatrix -= (m_LearningRate * deltaWeight) / Matrix::Map(gradSquaredW[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
			layer.BiasMatrix -= (m_LearningRate * deltaBias) / Matrix::Map(gradSquaredB[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
		}

		void Adadelta::Reset()
		{
			gradSquaredB.clear();
			gradSquaredW.clear();
		}
	}
}