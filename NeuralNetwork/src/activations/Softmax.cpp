#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		Matrix Softmax::Function(Matrix& x)
		{
			double sum = 0.0;
			Matrix::Map(x, [&sum](double a)
			{
				sum += exp(a); return a;
			});
			m_Activation = x.Map([sum](double a) { return exp(a) / sum; });
			return m_Activation;
		}

		Matrix Softmax::Derivative(Matrix& x)
		{
			return m_Activation.Map([](double a) { return a*(1 - a); });
		}

		Type Softmax::GetType() const
		{
			return SOFTMAX;
		}
	}
}