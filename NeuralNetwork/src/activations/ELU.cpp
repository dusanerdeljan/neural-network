#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		ELu::ELu(double alpha) : alpha(alpha)
		{
		}

		Matrix ELu::Function(Matrix& x)
		{
			return x.Map([alph=alpha](double a) { return a >= 0 ? a : alph*(exp(a) - 1); });
		}

		Matrix ELu::Derivative(Matrix& x)
		{
			return x.Map([alph=alpha](double a) { return a >= 0 ? 1 : alph*exp(a); });
		}

		Type ELu::GetType() const
		{
			return ELU;
		}
	}
}