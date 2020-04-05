#include "Optimizers.h"

namespace nn
{
	namespace optimizer
	{
		Adam::Adam(double lr, double beta1, double beta2) : Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2)
		{

		}

		void Adam::UpdateLayer(Layer& layer, Matrix& deltaWeight, Matrix& deltaBias, int layerIndex, unsigned int epoch)
		{
			Matrix firstUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());
			Matrix secondUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());

			Matrix firstUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());
			Matrix secondUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());

			if (firstMomentW.find(layerIndex) == firstMomentW.end())
			{
				// Weights
				firstMomentW[layerIndex] = (1 - m_Beta1) * deltaWeight;
				secondMomentW[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
				secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));

				// Biases
				firstMomentB[layerIndex] = (1 - m_Beta1) * deltaBias;
				secondMomentB[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
				secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));
			}
			else
			{
				// Weights
				firstMomentW[layerIndex] = firstMomentW[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaWeight;
				secondMomentW[layerIndex] = secondMomentW[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
				firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
				secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));

				// Biases
				firstMomentB[layerIndex] = firstMomentB[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaBias;
				secondMomentB[layerIndex] = secondMomentB[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
				secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));
			}

			Matrix deljenikW = Matrix::Map(secondUnbiasW, [](double x) { return sqrt(x) + 1e-7; });
			Matrix weight = (m_LearningRate * firstUnbiasW).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));

			Matrix deljenikB = Matrix::Map(secondUnbiasB, [](double x) { return sqrt(x) + 1e-7; });
			Matrix bias = (m_LearningRate * firstUnbiasB).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));

			layer.m_WeightMatrix -= weight;
			layer.m_BiasMatrix -= bias;
		}

		void Adam::Reset()
		{
			firstMomentW.clear();
			firstMomentB.clear();
			secondMomentW.clear();
			secondMomentB.clear();
		}
	}
}