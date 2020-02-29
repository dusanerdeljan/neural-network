#pragma once
#include "Matrix.h"
#include "ActivationFunctions.h"

class Layer
{
public:
	Matrix m_WeightMatrix;
	Matrix m_BiasMatrix;
	Activation::Type m_ActivationFunctionType;
	Activation::ActivationFunction* m_ActivationFunction;
	Matrix m_Activation;
	Matrix m_PreActivation;
public:
	template <typename... _Args> Layer(unsigned int inputNeurons, unsigned int outputNeurons, Activation::Type activationFunction, _Args... args);
	Matrix UpdateActivation(const Matrix& input);
	void SaveLayer(std::ofstream& outfile) const;
	static Layer LoadLayer(std::ifstream& infile);
};

template<typename... _Args>
Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, Activation::Type activationFunction, _Args... args)
	: m_WeightMatrix(outputNeurons, inputNeurons),
	m_BiasMatrix(outputNeurons, 1),
	m_ActivationFunctionType(activationFunction),
	m_Activation(outputNeurons, 1),
	m_ActivationFunction(ActivationFunctionFactory::BuildActivationFunction(activationFunction, args...)),
	m_PreActivation(outputNeurons, 1)
{

}

