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
