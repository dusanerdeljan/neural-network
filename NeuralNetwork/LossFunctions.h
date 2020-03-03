#pragma once
#include <numeric>
#include "Matrix.h"
#include "Layer.h"

namespace Loss{

	enum class Type
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
			Matrix gradient(layer.m_PreActivation);
			gradient.MapDerivative(layer.m_ActivationFunction);
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

	// TODO: Need to fix this later
	class CrossEntropy : public LossFunction
	{
	public:
		double GetLoss(const Matrix& prediction, const Matrix& target) const override
		{
			std::vector<double> predictionVector = prediction.GetColumnVector();
			std::vector<double> targetVector = target.GetColumnVector();
			std::vector<double> sumVector(predictionVector.size());
			std::transform(predictionVector.begin(), predictionVector.end(), targetVector.begin(), sumVector.begin(), [](double p, double t) { return t == 1 ? -log(p) : -log(1-p); });
			return std::accumulate(sumVector.begin(), sumVector.end(), 0.0);
		}
		Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
		{
			std::vector<double> predictionVector = prediction.GetColumnVector();
			std::vector<double> targetVector = target.GetColumnVector();
			std::vector<double> sumVector(predictionVector.size());
			std::transform(predictionVector.begin(), predictionVector.end(), targetVector.begin(), sumVector.begin(), [](double p, double t) { return t == 1 ? 1 / p : 1 / (1 - p); });
			return Matrix(sumVector);
		}
		Type GetType() const { return Type::CROSS_ENTROPY; }
	};

	class NegativeLogLikelihood : public LossFunction
	{
		double GetLoss(const Matrix& prediction, const Matrix& target) const override
		{
			std::vector<double> predictionVector = prediction.GetColumnVector();
			std::vector<double> targetVector = target.GetColumnVector();
			std::vector<double> sumVector(predictionVector.size());
			std::transform(predictionVector.begin() + 1, predictionVector.end(),
				targetVector.begin() + 1, sumVector.begin(),
				std::multiplies<double>());

			double suma = std::accumulate(sumVector.begin(), sumVector.end(), 0.0);
			return -log(suma);
		}
		Matrix GetDerivative(const Matrix& prediction, const Matrix& target) const override
		{
			std::vector<double> predictionVector = prediction.GetColumnVector();
			std::vector<double> targetVector = target.GetColumnVector();
			std::vector<double> sumVector(predictionVector.size());

			std::transform(predictionVector.begin() + 1, predictionVector.end(),
				targetVector.begin() + 1, sumVector.begin(),
				std::multiplies<double>());

			return Matrix::Map(Matrix(sumVector), [](double x) { return -1 / x; });
		}
		Type GetType() const { return Type::NLL; }
	};
}

class LossFunctionFactory
{
public:
	static Loss::LossFunction* BuildLossFunction(Loss::Type type)
	{
		switch (type)
		{
		case Loss::Type::MAE:
			return new Loss::MeanAbsoluteError();
		case Loss::Type::MSE:
			return new Loss::MeanSquaredError();
		case Loss::Type::QUADRATIC:
			return new Loss::Quadratic();
		case Loss::Type::HALF_QUADRATIC:
			return new Loss::HalfQuadratic();
		case Loss::Type::CROSS_ENTROPY:
			return new Loss::CrossEntropy();
		case Loss::Type::NLL:
			return new Loss::NegativeLogLikelihood();
		default:
			return nullptr;
		}
	}
};