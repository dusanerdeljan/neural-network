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