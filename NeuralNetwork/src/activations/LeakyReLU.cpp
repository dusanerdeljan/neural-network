/*
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

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