#pragma once
#include "Matrix.h"
#include "ActivationFunctions.h"

namespace nn
{
	class Layer
	{
	public:
		Matrix m_WeightMatrix;
		Matrix m_BiasMatrix;
		std::shared_ptr<activation::ActivationFunction> m_ActivationFunction;
		Matrix m_Activation;
		Matrix m_PreActivation;
	public:
		Layer(unsigned int inputNeurons, unsigned int outputNeurons, activation::Type activationFunction);
		Matrix UpdateActivation(const Matrix& input);
		void SaveLayer(std::ofstream& outfile) const;
		static Layer LoadLayer(std::ifstream& infile);
	};
}

