#include "WeightInitializers.h"

namespace nn
{
	namespace initialization
	{
		void Random::Initialize(Matrix& matrix) const
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(m_Min, m_Max);
			matrix.Map([&valueDistribution, &engine](double x)
			{
				return valueDistribution(engine);
			});
		}
	};
};