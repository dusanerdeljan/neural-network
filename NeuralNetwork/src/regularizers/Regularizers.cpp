#include "Regularizers.h"

namespace nn
{
	std::shared_ptr<regularizer::Regularizer> RegularizerFactory::BuildRegularizer(regularizer::Type type)
	{
		switch (type)
		{
		case regularizer::NONE:
			return std::make_shared<regularizer::Regularizer>();
		case regularizer::L1:
			return std::make_shared<regularizer::L1Regularizer>();
		case regularizer::L2:
			return std::make_shared<regularizer::L2Regularizer>();
		case regularizer::L1L2:
			return std::make_shared<regularizer::L1L2Regularizer>();
		default:
			return nullptr;
		}
	}
}
