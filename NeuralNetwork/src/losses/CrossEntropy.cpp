#include "LossFunctions.h"

namespace nn
{
	namespace loss
	{
		double CrossEntropy::GetLoss(const Matrix& prediction, const Matrix& target) const
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

		Matrix CrossEntropy::GetDerivative(const Matrix& prediction, const Matrix& target) const
		{
			return prediction - target;
		}

		Type CrossEntropy::GetType() const
		{
			return CROSS_ENTROPY;
		}
	}
}