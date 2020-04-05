#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		Matrix Tanh::Function(Matrix& x)
		{
			m_Activation = x.Map([](double a) { return (exp(a) - exp(-a)) / (exp(a) + exp(-a)); });
			return m_Activation;
		}

		Matrix Tanh::Derivative(Matrix& x)
		{
			return m_Activation.Map([](double a) { return 1 - pow(a, 2); });
		}

		Type Tanh::GetType() const
		{
			return TANH;
		}
	}
}