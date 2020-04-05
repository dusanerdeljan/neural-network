#include "Optimizers.h"

namespace nn
{
	namespace optimizer
	{
		Adagrad::Adagrad(double lr) : Optimizer(lr)
		{

		}

		void Adagrad::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			if (gradSquaredW.find(layerIndex) == gradSquaredW.end())
			{
				gradSquaredW[layerIndex] = Matrix::Map(deltaWeight, [](double x) { return x*x; });
				gradSquaredB[layerIndex] = Matrix::Map(deltaBias, [](double x) { return x*x; });
			}
			else
			{
				gradSquaredW[layerIndex] += Matrix::Map(deltaWeight, [](double x) { return x*x; });
				gradSquaredB[layerIndex] += Matrix::Map(deltaBias, [](double x) { return x*x; });
			}

			Matrix deljenikW = Matrix::Map(gradSquaredW[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
			Matrix deljenikB = Matrix::Map(gradSquaredB[layerIndex], [](double x) { return sqrt(x) + 1e-7; });

			layer.m_WeightMatrix -= (m_LearningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
			layer.m_BiasMatrix -= (m_LearningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));;
		}

		void Adagrad::Reset()
		{
			gradSquaredB.clear();
			gradSquaredW.clear();
		}
	}
}