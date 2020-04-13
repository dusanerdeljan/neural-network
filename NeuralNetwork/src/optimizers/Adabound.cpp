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
		Adabound::Adabound(double lr, double beta1, double beta2, double final_lr, double gamma)
			: Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2), m_FinalLearningRate(final_lr), m_Gamma(gamma)
		{

		}

		void Adabound::UpdateLayer(Layer & layer, Matrix & deltaWeight, Matrix & deltaBias, int layerIndex, unsigned int epoch)
		{
			double stepSize = m_LearningRate * (sqrt(1.0 - pow(m_Beta2, epoch)) / (1.0 - pow(m_Beta1, epoch)));
			double lowerBound = m_FinalLearningRate * (1.0 - 1.0 / (m_Gamma*epoch + 1.0));
			double upperBound = m_FinalLearningRate * (1.0 + 1.0 / (m_Gamma*epoch));
			if (msWeight.find(layerIndex) == msWeight.end())
			{
				msWeight[layerIndex] = (1.0 - m_Beta1)*deltaWeight;
				vsWeight[layerIndex] = (1.0 - m_Beta2)*Matrix::Map(deltaWeight, [](double x) { return x*x; });
				msBias[layerIndex] = (1.0 - m_Beta1)*deltaBias;
				vsBias[layerIndex] = (1.0 - m_Beta2)*Matrix::Map(deltaBias, [](double x) { return x*x; });
			}
			else
			{
				msWeight[layerIndex] = m_Beta1*msWeight[layerIndex] + (1.0 - m_Beta1)*deltaWeight;
				vsWeight[layerIndex] = m_Beta2*vsWeight[layerIndex] + (1.0 - m_Beta2)*Matrix::Map(deltaWeight, [](double x) { return x*x; });
				msBias[layerIndex] = m_Beta1*msBias[layerIndex] + (1.0 - m_Beta1)*deltaBias;
				vsBias[layerIndex] = m_Beta2*vsBias[layerIndex] + (1.0 - m_Beta2)*Matrix::Map(deltaBias, [](double x) { return x*x; });
			}
			Matrix boundedWeight = Matrix::Map(vsWeight[layerIndex], [stepSize, lowerBound, upperBound](double x)
			{
				double val = stepSize / (sqrt(x) + 1e-7);
				return std::min(std::max(val, lowerBound), upperBound);
			});
			Matrix boundedBias = Matrix::Map(vsBias[layerIndex], [stepSize, lowerBound, upperBound](double x)
			{
				double val = stepSize / (sqrt(x) + 1e-7);
				return std::min(std::max(val, lowerBound), upperBound);
			});
			layer.WeightMatrix -= msWeight[layerIndex].DotProduct(boundedWeight);
			layer.BiasMatrix -= msBias[layerIndex].DotProduct(boundedBias);
		}

		void Adabound::Reset()
		{
			msWeight.clear();
			vsWeight.clear();
			msBias.clear();
			vsBias.clear();
		}
	}
}