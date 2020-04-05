#include "WeightInitializers.h"

namespace nn
{
	std::shared_ptr<initialization::Initializer> WeightInitializerFactory::BuildWeightInitializer(initialization::Type type)
	{
		switch (type)
		{
		case initialization::RANDOM:
			return std::make_shared<initialization::Random>();
		case initialization::XAVIER_UNIFORM:
			return std::make_shared<initialization::XavierUniform>();
		case initialization::XAVIER_NORMAL:
			return std::make_shared<initialization::XavierNormal>();
		case initialization::HE_UNIFORM:
			return std::make_shared<initialization::HeUniform>();
		case initialization::HE_NORMAL:
			return std::make_shared<initialization::HeNormal>();
		case initialization::LECUN_UNIFORM:
			return std::make_shared<initialization::LeCunUniform>();
		case initialization::LECUN_NORMAL:
			return std::make_shared<initialization::LeCunNormal>();
		default:
			return nullptr;
		}
	}
};