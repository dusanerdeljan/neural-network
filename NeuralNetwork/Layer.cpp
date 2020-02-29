#include "Layer.h"

Matrix Layer::UpdateActivation(const Matrix & input)
{
	m_PreActivation = m_WeightMatrix*input + m_BiasMatrix;
	m_Activation = m_PreActivation;
	m_Activation.MapFunction(m_ActivationFunction);
	return m_Activation;
}

void Layer::SaveLayer(std::ofstream & outfile) const
{
	m_WeightMatrix.SaveMatrix(outfile);
	m_BiasMatrix.SaveMatrix(outfile);
	outfile.write((char*)&m_ActivationFunctionType, sizeof(m_ActivationFunctionType));
}

Layer Layer::LoadLayer(std::ifstream & infile)
{
	Matrix weightMatrix = Matrix::LoadMatrix(infile);
	Matrix biasMatrix = Matrix::LoadMatrix(infile);
	int activationType;
	infile.read((char*)&activationType, sizeof(activationType));
	Activation::Type type = Activation::Type(activationType);
	Layer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight(), type);
	layer.m_WeightMatrix = std::move(weightMatrix);
	layer.m_BiasMatrix = std::move(biasMatrix);
	return layer;
}

