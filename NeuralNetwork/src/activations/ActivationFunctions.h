#pragma once
#include <memory>
#include "../math/Matrix.h"

namespace nn
{
	namespace activation
	{
		enum Type
		{
			SIGMOID, RELU, LEAKY_RELU, ELU, TANH, SOFTMAX
		};

		class ActivationFunction
		{
		public:
			virtual Matrix Function(Matrix& x) = 0;
			virtual Matrix Derivative(Matrix& x) = 0;
			virtual Type GetType() const = 0;
			virtual void SaveActivationFunction(std::ofstream& out) const;
		};

		class Sigmoid : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class ReLu : public ActivationFunction
		{
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class LeakyReLu : public ActivationFunction
		{
		private:
			double alpha;
		public:
			LeakyReLu(double alpha = 0.1);
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class ELu : public ActivationFunction
		{
		private:
			double alpha;
		public:
			ELu(double alpha = 0.1);
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class Tanh : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class Softmax : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};
	}

	class ActivationFunctionFactory
	{
	public:
		static std::shared_ptr<activation::ActivationFunction> BuildActivationFunction(activation::Type type);
	};
}
