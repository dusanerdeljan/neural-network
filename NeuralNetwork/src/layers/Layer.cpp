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

#include "Layer.h"

namespace nn
{
	Matrix Layer::UpdateActivation(const Matrix & input)
	{
		WeightedSum = WeightMatrix*input + BiasMatrix;
		Activation = WeightedSum;
		Activation = ActivationFunction->Function(Activation);
		return Activation;
	}

	Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, nn::activation::Type activationFunction)
		: WeightMatrix(outputNeurons, inputNeurons),
		BiasMatrix(outputNeurons, 1),
		Activation(outputNeurons, 1),
		ActivationFunction(ActivationFunctionFactory::BuildActivationFunction(activationFunction)),
		WeightedSum(outputNeurons, 1)
	{

	}

	Layer::Layer(Layer && layer) noexcept : WeightMatrix(std::move(layer.WeightMatrix)), BiasMatrix(std::move(layer.BiasMatrix)),
		Activation(std::move(layer.Activation)), WeightedSum(std::move(layer.WeightedSum)), ActivationFunction(std::move(layer.ActivationFunction))
	{

	}

	Layer::Layer(const Layer & layer) : WeightMatrix(layer.WeightMatrix), BiasMatrix(layer.BiasMatrix),
		Activation(layer.Activation), WeightedSum(layer.WeightedSum), ActivationFunction(layer.ActivationFunction)
	{
	}

	void Layer::Initialize(const std::shared_ptr<initialization::Initializer> initializer)
	{
		initializer->Initialize(WeightMatrix);
	}

	void Layer::SaveLayer(std::ofstream & outfile) const
	{
		WeightMatrix.SaveMatrix(outfile);
		BiasMatrix.SaveMatrix(outfile);
		ActivationFunction->SaveActivationFunction(outfile);
	}

	Layer Layer::LoadLayer(std::ifstream & infile)
	{
		Matrix weightMatrix = Matrix::LoadMatrix(infile);
		Matrix biasMatrix = Matrix::LoadMatrix(infile);
		int activationType;
		infile.read((char*)&activationType, sizeof(activationType));
		activation::Type type = activation::Type(activationType);
		Layer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight(), type);
		layer.WeightMatrix = std::move(weightMatrix);
		layer.BiasMatrix = std::move(biasMatrix);
		return layer;
	}
	Layer & Layer::operator=(Layer && layer)
	{
		WeightMatrix = std::move(layer.WeightMatrix);
		BiasMatrix = std::move(layer.BiasMatrix);
		Activation = std::move(layer.Activation);
		ActivationFunction = std::move(layer.ActivationFunction);
		WeightedSum = std::move(layer.WeightedSum);
		return *this;
	}

}


