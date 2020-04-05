#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		double MeanAbsoluteError::GetLoss(const Matrix& prediction, const Matrix& target) const
		{
			return Matrix::Map(prediction - target, [](double x) { return abs(x); }).Sum();
		}

		Matrix MeanAbsoluteError::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return Matrix::Map(prediction - target, [](double x) { return x >= 0 ? 1 : -1; });
		}

		Type MeanAbsoluteError::GetType() const
		{
			return MAE;
		}
	}
}