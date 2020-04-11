#include "Optimizers.h"

namespace nn
{
	namespace optimizer
	{
		AMSGrad::AMSGrad(double lr, double beta1, double beta2) : Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2)
		{

		}

		void AMSGrad::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			if (firstMomentW.find(layerIndex) == firstMomentW.end())
			{
				// Weights
				firstMomentW[layerIndex] = (1 - m_Beta1) * deltaWeight;
				secondMomentW[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				infinityNormW[layerIndex] = secondMomentW[layerIndex];
				// Biases
				firstMomentB[layerIndex] = (1 - m_Beta1) * deltaBias;
				secondMomentB[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				infinityNormB[layerIndex] = secondMomentB[layerIndex];
			}
			else
			{
				// Weights
				firstMomentW[layerIndex] = firstMomentW[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaWeight;
				secondMomentW[layerIndex] = secondMomentW[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				infinityNormW[layerIndex] = Matrix::Max(infinityNormW[layerIndex], secondMomentW[layerIndex]);
				// Biases
				firstMomentB[layerIndex] = firstMomentB[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaBias;
				secondMomentB[layerIndex] = secondMomentB[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				infinityNormB[layerIndex] = Matrix::Max(infinityNormB[layerIndex], secondMomentB[layerIndex]);
			}
			layer.WeightMatrix -= m_LearningRate*firstMomentW[layerIndex] / (Matrix::Map(infinityNormW[layerIndex], [](double x) { return sqrt(x) + 1e-7; }));
			layer.BiasMatrix -= m_LearningRate*firstMomentB[layerIndex] / (Matrix::Map(infinityNormB[layerIndex], [](double x) { return sqrt(x) + 1e-7; }));
		}

		void AMSGrad::Reset()
		{
			firstMomentW.clear();
			firstMomentB.clear();
			secondMomentW.clear();
			secondMomentB.clear();
			infinityNormW.clear();
			infinityNormB.clear();
		}
	}
}