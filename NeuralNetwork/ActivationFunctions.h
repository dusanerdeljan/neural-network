#pragma once
#include <cmath>
#include <vector>

namespace Activation
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
			return y * (1 - y);
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
			return x >= 0 ? x : alpha * (exp(x) - 1);
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


// Moze i ovde da se uklopi, a mozemo i posebno loss functions namepsace napraviti
// Bice izmena verovatno kad startujemo sa nn i vidimo kako ovo zaista radi
// Verovatno postoje efikasniji nacini da se ovo sve implementira, to cemo kasnije
// -------------------------------------------------------------
class Softmax
{
public:
	std::vector<double> Function(std::vector<double>& x)
	{
		std::vector<double> output;
		double sum = 0;
		for (int i = 0; i < x.size(); ++i)
			sum += exp(x[i]);

		for (int i = 0; i < x.size(); ++i)
			output.push_back(x[i] / sum);

		return output;
	}

	std::vector<double> Derivative(std::vector<double>& x)
	{
		std::vector<double> y = Function(x);
		std::vector<double> output;
		for (int i = 0; i < x.size(); ++i)
			output.push_back(y[i] * (1 - y[i]));

		return output;
	}
};

class LogSoftmax
{
	std::vector<double> Function(std::vector<double>& x)
	{
		std::vector<double> output;
		double sum = 0;
		for (int i = 0; i < x.size(); ++i)
			sum += exp(x[i]);

		for (int i = 0; i < x.size(); ++i)
			output.push_back(log(sum) - x[i]);

		return output;
	}

	std::vector<double> Derivative(std::vector<double>& x)
	{
		// Need to be implemented
	}

};


// Negative Log-Likelihood
class NLL
{
	double Function(double x)
	{
		return -log(x);
	}

	double Derivative(double x)
	{
		return -1 / x;
	}
};

class CrossEntropy
{
	double Function(double x, double target)
	{
		return target == 1 ? -log(x) : -log(1 - x);
	}

	double Derivative(double x, double target)
	{
		return target == 1 ? -1 / x : 1 / (1 - x);
	}
};
// -------------------------------------------------------------
}


class ActivationFunctionFactory
{
public:
	template<typename... _Args>
	static Activation::ActivationFunction* BuildActivationFunction(Activation::Type type, _Args... args)
	{
		switch (type)
		{
		case Activation::Type::SIGMOID:
			return new Activation::Sigmoid();
		case Activation::Type::RELU:
			return new Activation::ReLu();
		case Activation::Type::LEAKY_RELU:
			return new Activation::LeakyReLu(args...);
		case Activation::Type::ELU:
			return new Activation::ELU(args...);
		case Activation::Type::TANH:
			return new Activation::Tanh();
		default:
			return nullptr;
		}
	}
};
