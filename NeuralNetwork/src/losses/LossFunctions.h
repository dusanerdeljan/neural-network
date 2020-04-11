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

#pragma once
#include <numeric>
#include "../layers/Layer.h"

namespace nn
{
	namespace loss
	{
		enum Type
		{
			MAE, MSE, QUADRATIC, HALF_QUADRATIC, CROSS_ENTROPY, NLL
		};

		class LossFunction
		{
		public:
			virtual double GetLoss(const Matrix& prediction, const Matrix& target) const = 0;
			virtual Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const = 0;
			virtual Type GetType() const = 0;
			Matrix Backward(Layer& layer, Matrix& error);
			void PropagateError(Layer& layer, Matrix& error) const;
		};

		class MeanAbsoluteError : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override;
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override;
			Type GetType() const override;
		};

		class MeanSquaredError : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override;
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override;
			Type GetType() const override;
		};

		class Quadratic : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override;
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override;
			Type GetType() const override;
		};

		class HalfQuadratic : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override;
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override;
			Type GetType() const override;
		};

		class CrossEntropy : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override;
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override;
			Type GetType() const override;
		};

		class NegativeLogLikelihood : public LossFunction
		{
			double GetLoss(const Matrix& prediction, const Matrix& target) const override;
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override;
			Type GetType() const override;
		};
	}

	class LossFunctionFactory
	{
	public:
		static std::shared_ptr<loss::LossFunction> BuildLossFunction(loss::Type type);
	};
}