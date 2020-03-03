#pragma once
#include "Matrix.h"
#include "ActivationFunctions.h"

class Layer
{
public:
	Matrix m_WeightMatrix;
	Matrix m_BiasMatrix;
	Activation::ActivationFunction* m_ActivationFunction;
	Matrix m_Activation;
	Matrix m_PreActivation;
public:
	Layer(unsigned int inputNeurons, unsigned int outputNeurons, Activation::ActivationFunction* activationFunction);
	Matrix UpdateActivation(const Matrix& input);
	void SaveLayer(std::ofstream& outfile) const;
	static Layer LoadLayer(std::ifstream& infile);
};

