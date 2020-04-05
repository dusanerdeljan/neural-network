#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		Matrix Sigmoid::Function(Matrix& x)
		{
			m_Activation = x.Map([](double a) { return 1 / (1 + exp(-a)); });
			return m_Activation;
		}

		Matrix Sigmoid::Derivative(Matrix& x)
		{
			return m_Activation.Map([](double a) { return a * (1 - a); });
		}
		Type Sigmoid::GetType() const
		{
			return SIGMOID;
		}
	}
}