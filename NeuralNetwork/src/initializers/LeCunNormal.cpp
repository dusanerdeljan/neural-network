#include "WeightInitializers.h"

namespace nn
{
	namespace initialization
	{
		void LeCunNormal::Initialize(Matrix& matrix) const
		{
			std::default_random_engine engine;
			std::normal_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 1.0 / sqrt(matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return valueDistribution(engine) * factor;
			});
		}
	}
};