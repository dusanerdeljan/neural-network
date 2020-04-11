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