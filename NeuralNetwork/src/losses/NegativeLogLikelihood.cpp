#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		double NegativeLogLikelihood::GetLoss(const Matrix& prediction, const Matrix& target) const
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

		Matrix NegativeLogLikelihood::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return prediction - target;
		}

		Type NegativeLogLikelihood::GetType() const
		{
			return NLL;
		}
	}
}