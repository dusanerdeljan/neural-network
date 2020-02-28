#include "Layer.h"

Matrix Layer::UpdateActivation(const Matrix & input)
{
	m_PreActivation = m_WeightMatrix*input + m_BiasMatrix;
	m_Activation = m_PreActivation;
	m_Activation.MapFunction(m_ActivationFunction);
	return m_Activation;
}

