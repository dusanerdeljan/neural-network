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

#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		double MeanSquaredError::GetLoss(const Matrix& prediction, const Matrix& target) const
		{
			return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum() / (target.GetWidth() * target.GetHeight());
		}

		Matrix MeanSquaredError::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return (prediction - target) * (2.0 / (target.GetWidth() * target.GetHeight()));
		}

		Type MeanSquaredError::GetType() const
		{
			return MSE;
		}
	}
}