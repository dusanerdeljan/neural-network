#include "Optimizers.h"

namespace nn
{
	namespace optimizer
	{
		GradientDescent::GradientDescent(double lr) : Optimizer(lr)
		{

		}

		void GradientDescent::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			layer.m_WeightMatrix -= m_LearningRate * deltaWeight;
			layer.m_BiasMatrix -= m_LearningRate * deltaBias;
		}
	}
}