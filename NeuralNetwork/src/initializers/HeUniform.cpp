#include "WeightInitializers.h"

namespace nn
{
	namespace initialization
	{
		void HeUniform::Initialize(Matrix& matrix) const
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(6.0 / matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return (valueDistribution(engine) - 0.5) * factor;
			});
		}
	};
};