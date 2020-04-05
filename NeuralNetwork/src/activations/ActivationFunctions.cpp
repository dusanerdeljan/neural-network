#include "ActivationFunctions.h"

namespace nn
{
	namespace activation
	{
		void ActivationFunction::SaveActivationFunction(std::ofstream & out) const
		{
			Type type = GetType();
			out.write((char*)&type, sizeof(type));
		}
	}

	std::shared_ptr<activation::ActivationFunction> ActivationFunctionFactory::BuildActivationFunction(activation::Type type)
	{
		switch (type)
		{
		case activation::Type::SIGMOID:
			return std::make_shared<activation::Sigmoid>();
		case activation::Type::RELU:
			return std::make_shared<activation::ReLu>();
		case activation::Type::LEAKY_RELU:
			return std::make_shared<activation::LeakyReLu>();
		case activation::Type::ELU:
			return std::make_shared<activation::ELu>();
		case activation::Type::TANH:
			return std::make_shared<activation::Tanh>();
		case activation::Type::SOFTMAX:
			return std::make_shared<activation::Softmax>();
		default:
			return nullptr;
		}
	}
}