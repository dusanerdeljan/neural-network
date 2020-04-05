#include "Regularizers.h"

namespace nn
{
	namespace regularizer
	{
		void L2Regularizer::Regularize(const Matrix& weights, Matrix& gradient) const
		{
			gradient += 2 * m_L2*weights;
		}
	}
}