#include "Layer.h"

Matrix Layer::UpdateActivation(const Matrix & input)
{
	m_PreActivation = m_WeightMatrix*input + m_BiasMatrix;
	m_Activation = m_PreActivation;
	m_Activation = m_ActivationFunction->Function(m_PreActivation);
	return m_Activation;
}

Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, Activation::ActivationFunction* activationFunction)
	: m_WeightMatrix(outputNeurons, inputNeurons),
	m_BiasMatrix(outputNeurons, 1),
	m_Activation(outputNeurons, 1),
	m_ActivationFunction(activationFunction),
	m_PreActivation(outputNeurons, 1)
{

}

void Layer::SaveLayer(std::ofstream & outfile) const
{
	m_WeightMatrix.SaveMatrix(outfile);
	m_BiasMatrix.SaveMatrix(outfile);
	m_ActivationFunction->SaveActivationFunction(outfile);
}

Layer Layer::LoadLayer(std::ifstream & infile)
{
	Matrix weightMatrix = Matrix::LoadMatrix(infile);
	Matrix biasMatrix = Matrix::LoadMatrix(infile);
	int activationType;
	infile.read((char*)&activationType, sizeof(activationType));
	Activation::Type type = Activation::Type(activationType);
	Layer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight(), ActivationFunctionFactory::BuildActivationFunction(type));
	layer.m_WeightMatrix = std::move(weightMatrix);
	layer.m_BiasMatrix = std::move(biasMatrix);
	return layer;
}

