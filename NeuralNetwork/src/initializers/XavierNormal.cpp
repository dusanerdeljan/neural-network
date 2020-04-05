#include "WeightInitializers.h"

namespace nn
{
	namespace initialization
	{
		void XavierNormal::Initialize(Matrix& matrix) const
		{
			std::default_random_engine engine;
			std::normal_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(6.0 / (matrix.GetWidth() + matrix.GetHeight()));
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return valueDistribution(engine) * factor;
			});
		}
	}
};