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
		double NegativeLogLikelihood::GetLoss(const Matrix& prediction, const Matrix& target) const
		{
			std::vector<double> predictionVector = prediction.GetColumnVector();
			std::vector<double> targetVector = target.GetColumnVector();
			double sum = 0.0;
			std::vector<double>::iterator tIt = targetVector.begin();
			for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
			{
				double value = *tIt*log(*pIt);
				if (!isnan(value)) sum -= value;
			}
			return sum;
		}

		Matrix NegativeLogLikelihood::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return prediction - target;
		}

		Type NegativeLogLikelihood::GetType() const
		{
			return NLL;
		}
	}
}