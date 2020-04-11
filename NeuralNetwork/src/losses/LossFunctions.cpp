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
		Matrix LossFunction::Backward(Layer & layer, Matrix & error)
		{
			Matrix gradient = layer.ActivationFunction->Derivative(layer.WeightedSum);
			gradient.DotProduct(error);
			return gradient;
		}
		void LossFunction::PropagateError(Layer & layer, Matrix & error) const
		{
			error = Matrix::Transpose(layer.WeightMatrix) * error;
		}
	}

	std::shared_ptr<loss::LossFunction> LossFunctionFactory::BuildLossFunction(loss::Type type)
	{
		switch (type)
		{
		case loss::Type::MAE:
			return std::make_shared<loss::MeanAbsoluteError>();
		case loss::Type::MSE:
			return std::make_shared<loss::MeanSquaredError>();
		case loss::Type::QUADRATIC:
			return std::make_shared<loss::Quadratic>();
		case loss::Type::HALF_QUADRATIC:
			return std::make_shared<loss::HalfQuadratic>();
		case loss::Type::CROSS_ENTROPY:
			return std::make_shared<loss::CrossEntropy>();
		case loss::Type::NLL:
			return std::make_shared<loss::NegativeLogLikelihood>();
		default:
			return nullptr;
		}
	}
};