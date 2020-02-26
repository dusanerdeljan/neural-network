#pragma once
#include "Matrix.h"
#include "ActivationFunctions.h"

class Layer
{
public:
	Matrix m_WeightMatrix;
	Matrix m_BiasMatrix;
	ActivationFunctions::Type m_ActivationFunctionType;
	Matrix m_Activation;
public:
	Layer(unsigned int inputNeurons, unsigned int outputNeurons, ActivationFunctions::Type activationFunction);
	Matrix UpdateActivation(const Matrix& input);
	~Layer();
};

