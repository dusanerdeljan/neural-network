#pragma once
#include <numeric>
#include "Matrix.h"
#include "Layer.h"

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
			Matrix Backward(Layer& layer, Matrix& error) const
			{
				Matrix gradient = layer.m_ActivationFunction->Derivative(layer.m_PreActivation);
				gradient.DotProduct(error);
				return gradient;
			}
			void PropagateError(Layer& layer, Matrix& error) const
			{
				error = Matrix::Transpose(layer.m_WeightMatrix) * error;
			}
		};

		class MeanAbsoluteError : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override
			{
				return Matrix::Map(prediction - target, [](double x) { return abs(x); }).Sum();
			}
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
			{
				return Matrix::Map(prediction - target, [](double x) { return x >= 0 ? 1 : -1; });
			}
			Type GetType() const { return Type::MAE; }
		};

		class MeanSquaredError : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override
			{
				return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum() / (target.GetWidth() * target.GetHeight());
			}
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
			{
				return (prediction - target) * (2.0 / (target.GetWidth() * target.GetHeight()));
			}
			Type GetType() const { return Type::MSE; }
		};

		class Quadratic : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override
			{
				return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum();
			}
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
			{
				return (prediction - target) * 2.0;
			}
			Type GetType() const { return Type::QUADRATIC; }
		};

		class HalfQuadratic : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override
			{
				return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum() / 2.0;
			}
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
			{
				return (prediction - target);
			}
			Type GetType() const { return Type::HALF_QUADRATIC; }
		};

		class CrossEntropy : public LossFunction
		{
		public:
			double GetLoss(const Matrix& prediction, const Matrix& target) const override
			{
				std::vector<double> predictionVector = prediction.GetColumnVector();
				std::vector<double> targetVector = target.GetColumnVector();
				double sum = 0.0;
				std::vector<double>::iterator tIt = targetVector.begin();
				for (std::vector<double>::iterator pIt = predictionVector.begin(); pIt != predictionVector.end(); ++pIt, ++tIt)
				{
					double value = -*tIt*log(*pIt) - (1 - *tIt)*log(1 - *pIt);
					if (!isnan(value)) sum += value;
				}
				return sum;
			}
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
			{
				return prediction - target;
			}
			Type GetType() const { return Type::CROSS_ENTROPY; }
		};

		class NegativeLogLikelihood : public LossFunction
		{
			double GetLoss(const Matrix& prediction, const Matrix& target) const override
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
			Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
			{
				return prediction - target;
			}
			Type GetType() const { return Type::NLL; }
		};
	}

	class LossFunctionFactory
	{
	public:
		static std::shared_ptr<loss::LossFunction> BuildLossFunction(loss::Type type)
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
}