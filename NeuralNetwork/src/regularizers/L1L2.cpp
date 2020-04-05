#include "Regularizers.h"

namespace nn
{
	namespace regularizer
	{
		void L1L2Regularizer::Regularize(const Matrix& weights, Matrix& gradient) const
		{
			gradient += Matrix::Map(weights, [l1 = m_L1, l2 = m_L2](double x)
			{
				double sign = x >= 0 ? l1 : -l1;
				return sign * 2 * l2*x;
			});
		}
	}
}