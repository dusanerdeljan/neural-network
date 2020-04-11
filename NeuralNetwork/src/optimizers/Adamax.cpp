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
		Adamax::Adamax(double lr, double beta1, double beta2) : Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2)
		{

		}

		void Adamax::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			if (firstMomentW.find(layerIndex) == firstMomentW.end())
			{
				firstMomentW[layerIndex] = (1 - m_Beta1)*deltaWeight;
				infinityNormW[layerIndex] = Matrix::Map(deltaWeight, [](double x) { return abs(x); });
				firstMomentB[layerIndex] = (1 - m_Beta1)*deltaBias;
				infinityNormB[layerIndex] = Matrix::Map(deltaBias, [](double x) { return abs(x); });
			}
			else
			{
				firstMomentW[layerIndex] = m_Beta1*firstMomentW[layerIndex] + (1 - m_Beta1)*deltaWeight;
				infinityNormW[layerIndex] = Matrix::Max(m_Beta2*infinityNormW[layerIndex], Matrix::Map(deltaWeight, [](double x) { return abs(x); }));
				firstMomentB[layerIndex] = m_Beta1*firstMomentB[layerIndex] + (1 - m_Beta1)*deltaBias;
				infinityNormB[layerIndex] = Matrix::Max(m_Beta2*infinityNormB[layerIndex], Matrix::Map(deltaBias, [](double x) { return abs(x); }));
			}
			double lr_t = m_LearningRate / (1 - pow(m_Beta1, epoch));
			layer.WeightMatrix -= lr_t * firstMomentW[layerIndex] / (Matrix::Map(infinityNormW[layerIndex], [](double x) { return x + 1e-7; }));
			layer.BiasMatrix -= lr_t * firstMomentB[layerIndex] / (Matrix::Map(infinityNormB[layerIndex], [](double x) { return x + 1e-7; }));
		}

		void Adamax::Reset()
		{
			firstMomentW.clear();
			firstMomentB.clear();
			infinityNormW.clear();
			infinityNormB.clear();
		}
	}
}