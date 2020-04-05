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
			layer.m_WeightMatrix += -m_Momentum*previousWeight + (1 + m_Momentum)*lastMomentWeight[layerIndex];
			layer.m_BiasMatrix += -m_Momentum*previousBias + (1 + m_Momentum)*lastMomentBias[layerIndex];
		}

		void Nesterov::Reset()
		{
			lastMomentWeight.clear();
			lastMomentBias.clear();
		}
	}
}