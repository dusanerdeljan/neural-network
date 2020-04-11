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

#pragma once
#include "../math/Matrix.h"
#include "../activations/ActivationFunctions.h"
#include "../initializers/WeightInitializers.h"

namespace nn
{
	class Layer
	{
	public:
		Matrix WeightMatrix;
		Matrix BiasMatrix;
		std::shared_ptr<activation::ActivationFunction> ActivationFunction;
		Matrix Activation;
		Matrix WeightedSum;
	public:
		Layer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
		void Initialize(const std::shared_ptr<initialization::Initializer> initializer);
		Matrix UpdateActivation(const Matrix& input);
		void SaveLayer(std::ofstream& outfile) const;
		static Layer LoadLayer(std::ifstream& infile);
		Layer& operator=(Layer&& layer);
		Layer(Layer&& layer) noexcept;
		Layer(const Layer& layer);
	};
}

