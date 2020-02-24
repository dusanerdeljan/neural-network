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
			double y = Function(x);
			return y*(1 - y);
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
			if (x >= value)
				return x;
			else
				return  value;
		}

		double Derivative(double x) override
		{
			return x >= 0 ? 1 : alpha;
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
			if (x >= 0)
				return x;
			else
				return alpha * (exp(x) - 1);
		}

		double Derivative(double x) override
		{
			return x < 0 ? Function(x) + alpha : 1;
		}
	};

	class Tanh : public ActivationFunction
	{
	public:
		double Function(double x) override
		{
			return 2 / (1 + exp(-2 * x)) - 1;
		}

		double Derivative(double x) override
		{
			double y = Function(x);
			return 1 - y*y;
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
