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