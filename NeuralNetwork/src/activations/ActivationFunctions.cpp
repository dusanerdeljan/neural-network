/*
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

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