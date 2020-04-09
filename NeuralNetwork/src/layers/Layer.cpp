#include "Layer.h"

namespace nn
{
	Matrix Layer::UpdateActivation(const Matrix & input)
	{
		m_PreActivation = m_WeightMatrix*input + m_BiasMatrix;
		m_Activation = m_PreActivation;
		m_Activation = m_ActivationFunction->Function(m_Activation);
		return m_Activation;
	}

	Layer::Layer(unsigned int inputNeurons, unsigned int outputNeurons, nn::activation::Type activationFunction)
		: m_WeightMatrix(outputNeurons, inputNeurons),
		m_BiasMatrix(outputNeurons, 1),
		m_Activation(outputNeurons, 1),
		m_ActivationFunction(ActivationFunctionFactory::BuildActivationFunction(activationFunction)),
		m_PreActivation(outputNeurons, 1)
	{

	}

	Layer::Layer(Layer && layer) noexcept : m_WeightMatrix(std::move(layer.m_WeightMatrix)), m_BiasMatrix(std::move(layer.m_BiasMatrix)),
		m_Activation(std::move(layer.m_Activation)), m_PreActivation(std::move(layer.m_PreActivation)), m_ActivationFunction(std::move(layer.m_ActivationFunction))
	{

	}

	Layer::Layer(const Layer & layer) : m_WeightMatrix(layer.m_WeightMatrix), m_BiasMatrix(layer.m_BiasMatrix),
		m_Activation(layer.m_Activation), m_PreActivation(layer.m_PreActivation), m_ActivationFunction(layer.m_ActivationFunction)
	{
	}

	void Layer::Initialize(const std::shared_ptr<initialization::Initializer> initializer)
	{
		initializer->Initialize(m_WeightMatrix);
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
		activation::Type type = activation::Type(activationType);
		Layer layer(weightMatrix.GetWidth(), weightMatrix.GetHeight(), type);
		layer.m_WeightMatrix = std::move(weightMatrix);
		layer.m_BiasMatrix = std::move(biasMatrix);
		return layer;
	}
	Layer & Layer::operator=(Layer && layer)
	{
		m_WeightMatrix = std::move(layer.m_WeightMatrix);
		m_BiasMatrix = std::move(layer.m_BiasMatrix);
		m_Activation = std::move(layer.m_Activation);
		m_ActivationFunction = std::move(layer.m_ActivationFunction);
		m_PreActivation = std::move(layer.m_PreActivation);
		return *this;
	}

}


