#include "Layer.h"

double SigmoidFunction(double x) { return 1 / (1 + exp(-x)); }

Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, ActivationFunctions::Type activationFunction)
	: m_WeightMatrix(outputNeurons, inputNeurons), m_BiasMatrix(outputNeurons, 1), m_ActivationFunctionType(activationFunction), m_Activation(outputNeurons, 1)
{
	m_WeightMatrix.Randomize();
	m_BiasMatrix.Randomize();
}

Matrix Layer::UpdateActivation(const Matrix & input)
{
	m_Activation = m_WeightMatrix*input + m_BiasMatrix;
	m_Activation.Map(SigmoidFunction); // Hard coded for now
	return m_Activation;
}

Layer::~Layer()
{
}
