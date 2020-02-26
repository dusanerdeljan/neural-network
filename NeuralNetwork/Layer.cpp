#include "Layer.h"

double SigmoidFunction(double x) { return 1 / (1 + exp(-x)); }

Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, ActivationFunctions::Type activationFunction)
	: m_WeightMatrix(inputNeurons, outputNeurons), m_BiasMatrix(1, outputNeurons), m_ActivationFunctionType(activationFunction), m_Activation(1, outputNeurons)
{

}

Matrix Layer::UpdateActivation(const Matrix & input)
{
	m_Activation = input*m_WeightMatrix + m_BiasMatrix;
	m_Activation.Map(SigmoidFunction); // Hard coded for now
	return m_Activation;
}

Layer::~Layer()
{
}
