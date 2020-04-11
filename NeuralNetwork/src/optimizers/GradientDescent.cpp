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
			layer.WeightMatrix -= m_LearningRate * deltaWeight;
			layer.BiasMatrix -= m_LearningRate * deltaBias;
		}
	}
}