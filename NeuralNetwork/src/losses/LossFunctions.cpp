#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		Matrix LossFunction::Backward(Layer & layer, Matrix & error)
		{
			Matrix gradient = layer.m_ActivationFunction->Derivative(layer.m_PreActivation);
			gradient.DotProduct(error);
			return gradient;
		}
		void LossFunction::PropagateError(Layer & layer, Matrix & error) const
		{
			error = Matrix::Transpose(layer.m_WeightMatrix) * error;
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