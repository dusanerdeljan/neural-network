#pragma once
#include <cmath>

namespace ActivationFunctions
{
	enum class Type
	{
		SIGMOID, RELU, LEAKY_RELU, ELU, TANH
	};

	class ActivationFunction
	{
	public:
		virtual double Function(double x) = 0;
		virtual double Derivative(double x) = 0;
	};

	class Sigmoid : public ActivationFunction
	{
	public:
		double Function(double x) override
		{
			return 1 / (1 + exp(-x));
		}

		double Derivative(double x) override
		{
			return Function(x) * (1 - Function(x));
		}
	};

	class ReLu : public ActivationFunction
	{
	public:
		double Function(double x) override
		{
			return x >= 0 ? x : 0;
		}

		double Derivative(double x) override
		{
			return x >= 0 ? 1 : 0;
		}
	};

	class LeakyReLu : public ActivationFunction
	{
	private:
		double alpha;
	public:
		LeakyReLu(double alpha = 0.1) : alpha(alpha) {}
		double Function(double x) override
		{
			double value = alpha*x;
			return x >= value ? x : value;
		}

		double Derivative(double x) override
		{
			double value = alpha*x;
			return x >= value ? 1 : 0;
		}
	};

	class ELU : public ActivationFunction
	{
	private:
		double alpha;
	public:
		ELU(double alpha = 0.1) : alpha(alpha) {}
		double Function(double x) override
		{
			return x >= 0 ? x : alpha * (exp(x) - 1)
		}

		double Derivative(double x) override
		{
			return x >= 0 ? 1 : alpha*exp(x);
		}
	};

	class Tanh : public ActivationFunction
	{
	public:
		double Function(double x) override
		{
			return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
		}

		double Derivative(double x) override
		{
			return 1 - pow(Function(x), 2);
		}
	};
}

class ActivationFunctionFactory
{
public:
	template<typename... _Args>
	static ActivationFunctions::ActivationFunction* BuildActivationFunction(ActivationFunctions::Type type, _Args... args)
	{
		switch (type)
		{
		case ActivationFunctions::Type::SIGMOID:
			return new ActivationFunctions::Sigmoid();
		case ActivationFunctions::Type::RELU:
			return new ActivationFunctions::ReLu();
		case ActivationFunctions::Type::LEAKY_RELU:
			return new ActivationFunctions::LeakyReLu(args...);
		case ActivationFunctions::Type::ELU:
			return new ActivationFunctions::ELU(args...);
		case ActivationFunctions::Type::TANH:
			return new ActivationFunctions::Tanh();
		default:
			return nullptr;
		}
	}
};
