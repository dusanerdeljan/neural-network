#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		double Quadratic::GetLoss(const Matrix& prediction, const Matrix& target) const
		{
			return Matrix::Map(prediction - target, [](double x) { return x*x; }).Sum();
		}

		Matrix Quadratic::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return (prediction - target) * 2.0;
		}

		Type Quadratic::GetType() const
		{
			return QUADRATIC;
		}
	}
}