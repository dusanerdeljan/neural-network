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
		Nadam::Nadam(double lr, double beta1, double beta2) : Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2)
		{

		}

		void Nadam::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			if (firstMomentW.find(layerIndex) == firstMomentW.end())
			{
				// Weights
				firstMomentW[layerIndex] = (1 - m_Beta1) * deltaWeight;
				secondMomentW[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				// Biases
				firstMomentB[layerIndex] = (1 - m_Beta1) * deltaBias;
				secondMomentB[layerIndex] = Matrix::Map(deltaBias, [](double x) { return abs(x); });
			}
			else
			{
				// Weights
				firstMomentW[layerIndex] = firstMomentW[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaWeight;
				secondMomentW[layerIndex] = secondMomentW[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				// Biases
				firstMomentB[layerIndex] = firstMomentB[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaBias;
				secondMomentB[layerIndex] = secondMomentB[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
			}

			Matrix firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
			Matrix secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));
			Matrix firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
			Matrix secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));

			layer.WeightMatrix -= (m_LearningRate * (firstUnbiasW * m_Beta1 + (1 - m_Beta1) / (1 - pow(m_Beta1, epoch)) * deltaWeight)) / Matrix::Map(secondUnbiasW, [](double x) { return sqrt(x) + 1e-7; });
			layer.BiasMatrix -= (m_LearningRate * (firstUnbiasB * m_Beta1 + (1 - m_Beta1) / (1 - pow(m_Beta1, epoch)) * deltaBias)) / Matrix::Map(secondUnbiasB, [](double x) { return sqrt(x) + 1e-7; });
		}

		void Nadam::Reset()
		{
			firstMomentW.clear();
			firstMomentB.clear();
			secondMomentW.clear();
			secondMomentB.clear();
		}
	}
}