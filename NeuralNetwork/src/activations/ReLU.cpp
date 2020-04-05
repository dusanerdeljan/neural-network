#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		Matrix ReLu::Function(Matrix& x)
		{
			return x.Map([](double a) { return a >= 0 ? a : 0; });
		}

		Matrix ReLu::Derivative(Matrix& x)
		{
			return x.Map([](double a) { return a >= 0 ? 1 : 0; });
		}

		Type ReLu::GetType() const
		{
			return RELU;
		}
	}
}