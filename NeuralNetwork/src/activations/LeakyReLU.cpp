#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		LeakyReLu::LeakyReLu(double alpha) : alpha(alpha)
		{
		}

		Matrix LeakyReLu::Function(Matrix& x)
		{
			return x.Map([alph=alpha](double a) { return a >= alph*a ? a : alph; });
		}

		Matrix LeakyReLu::Derivative(Matrix& x)
		{
			return x.Map([alph=alpha](double a) { return a >= alph*a ? 1 : 0; });
		}

		Type LeakyReLu::GetType() const
		{
			return LEAKY_RELU;
		}
	}
}