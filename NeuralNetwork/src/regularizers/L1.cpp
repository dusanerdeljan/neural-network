#include "Regularizers.h"

namespace nn
{
	namespace regularizer
	{
		void L1Regularizer::Regularize(const Matrix& weights, Matrix& gradient) const
		{
			gradient += Matrix::Map(weights, [l1 = m_L1](double x) { return x >= 0 ? l1 : -l1; });
		}
	}
}