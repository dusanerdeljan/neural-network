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