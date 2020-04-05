#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		double HalfQuadratic::GetLoss(const Matrix& prediction, const Matrix& target) const
		{
			return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum() / 2.0;
		}

		Matrix HalfQuadratic::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return (prediction - target);
		}

		Type HalfQuadratic::GetType() const
		{
			return HALF_QUADRATIC;
		}
	}
}