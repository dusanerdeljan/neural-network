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
			layer.m_WeightMatrix -= m_LearningRate * lastDeltaWeight[layerIndex];
			layer.m_BiasMatrix -= m_LearningRate * lastDeltaBias[layerIndex];
		}

		void Momentum::Reset()
		{
			lastDeltaWeight.clear();
			lastDeltaBias.clear();
		}
	}
}