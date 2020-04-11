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
			Matrix deljenikW = Matrix::Map(gradSquaredW[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
			Matrix deljenikB = Matrix::Map(gradSquaredB[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
			layer.WeightMatrix -= (m_LearningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
			layer.BiasMatrix -= (m_LearningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));
		}

		void Adadelta::Reset()
		{
			gradSquaredB.clear();
			gradSquaredW.clear();
		}
	}
}