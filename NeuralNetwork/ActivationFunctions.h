#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <unordered_map>
#include "Matrix.h"

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
			virtual inline Type GetType() const = 0;
			virtual void SaveActivationFunction(std::ofstream& out) const
			{
				Type type = GetType();
				out.write((char*)&type, sizeof(type));
			}
		};

		class Sigmoid : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override
			{
				//return 1 / (1 + exp(-x));
				m_Activation = x.Map([](double a) { return 1 / (1 + exp(-a)); });
				return m_Activation;
			}

			Matrix Derivative(Matrix& x) override
			{
				return m_Activation.Map([](double a) { return a * (1 - a); });
			}

			inline Type GetType() const override { return Type::SIGMOID; }
		};

		class ReLu : public ActivationFunction
		{
		public:
			Matrix Function(Matrix& x) override
			{
				//return x >= 0 ? x : 0;
				return x.Map([](double a) { return a >= 0 ? a : 0; });
			}

			Matrix Derivative(Matrix& x) override
			{
				//return x >= 0 ? 1 : 0;
				return x.Map([](double a) { return a >= 0 ? 1 : 0; });
			}

			inline Type GetType() const override { return Type::RELU; }
		};

		class LeakyReLu : public ActivationFunction
		{
		private:
			double alpha;
		public:
			LeakyReLu(double alpha = 0.1) : alpha(alpha) {}
			Matrix Function(Matrix& x) override
			{
				//double value = alpha*x;
				//return x >= value ? x : value;
				double alph = alpha;
				return x.Map([alph](double a) { return a >= alph*a ? a : alph; });
			}

			Matrix Derivative(Matrix& x) override
			{
				//double value = alpha*x;
				//return x >= value ? 1 : 0;
				double alph = alpha;
				return x.Map([alph](double a) { return a >= alph*a ? 1 : 0; });
			}

			inline Type GetType() const override { return Type::LEAKY_RELU; }
		};

		class ELu : public ActivationFunction
		{
		private:
			double alpha;
		public:
			ELu(double alpha = 0.1) : alpha(alpha) {}
			Matrix Function(Matrix& x) override
			{
				//return x >= 0 ? x : alpha * (exp(x) - 1);
				double alph = alpha;
				return x.Map([alph](double a) { return a >= 0 ? a : alph*(exp(a) - 1); });
			}

			Matrix Derivative(Matrix& x) override
			{
				//return x >= 0 ? 1 : alpha*exp(x);
				double alph = alpha;
				return x.Map([alph](double a) { return a >= 0 ? 1 : alph*exp(a); });
			}

			inline Type GetType() const override { return Type::ELU; }
		};

		class Tanh : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override
			{
				//return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
				m_Activation = x.Map([](double a) { return (exp(a) - exp(-a)) / (exp(a) + exp(-a)); });
				return m_Activation;
			}

			Matrix Derivative(Matrix& x) override
			{
				//return 1 - pow(Function(x), 2);
				return m_Activation.Map([](double a) { return 1 - pow(a, 2); });
			}

			inline Type GetType() const override { return Type::TANH; }
		};

		class Softmax : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x)
			{
				//std::vector<double> output;
				//double sum = 0;
				//for (unsigned int i = 0; i < x.size(); ++i)
				//	sum += exp(x[i]);

				//for (unsigned int i = 0; i < x.size(); ++i)
				//	output.push_back(exp(x[i]) / sum);

				//return output;
				double sum = 0.0;
				Matrix::Map(x, [&sum](double a)
				{
					sum += exp(a); return a;
				});
				m_Activation = x.Map([sum](double a) { return exp(a) / sum; });
				return m_Activation;
			}

			Matrix Derivative(Matrix& x)
			{
				//std::vector<double> y = Function(x);
				//std::vector<double> output;
				//for (unsigned int i = 0; i < x.size(); ++i)
				//	output.push_back(y[i] * (1 - y[i]));

				//return output;
				return m_Activation.Map([](double a) { return a*(1 - a); });
			}

			inline Type GetType() const override { return Type::SOFTMAX; }
		};
	}

	class ActivationFunctionFactory
	{
	public:
		static std::shared_ptr<activation::ActivationFunction> BuildActivationFunction(activation::Type type)
		{
			switch (type)
			{
			case activation::Type::SIGMOID:
				return std::make_shared<activation::Sigmoid>();
			case activation::Type::RELU:
				return std::make_shared<activation::ReLu>();
			case activation::Type::LEAKY_RELU:
				return std::make_shared<activation::LeakyReLu>();
			case activation::Type::ELU:
				return std::make_shared<activation::ELu>();
			case activation::Type::TANH:
				return std::make_shared<activation::Tanh>();
			case activation::Type::SOFTMAX:
				return std::make_shared<activation::Softmax>();
			default:
				return nullptr;
			}
		}
	};
}
