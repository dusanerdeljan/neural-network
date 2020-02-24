#pragma once
#include <cmath>

namespace ActivationFunctions
{
	enum class Type
	{
		SIGMOID, RELU, LEAKY_RELU, ELU, TANH
	};
	// Need to build interface for activation functons
	// -----------------------------------------------------

	// Ako aktivaciona funkcija ima jos neki parametar osim x moze se sacuvati kao atribut pa da se napravi neka privatna funkcija
	// koja ca se pozivati u Function/Derivative sa tim dodatnim parametrom
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
			// Should be implemented
			return 0.0;
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
			// Should be implemented
			return 0.0;
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
			// Should be implemented
			return 0.0;
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
			// Should be implemented
			return 0.0;
		}
	};

	class Tanh : public ActivationFunction
	{
	public:
		double Function(double x) override
		{
			return 2 / 1 + exp(-2 * x) - 1;
		}

		double Derivative(double x) override
		{
			// Should be implemented
			return 0.0;
		}
	};
	// -----------------------------------------------------
}

// Kao dodatni parametar mogu da se proslede parametri koji ce biti prosledjeni u konstruktor odgovarajuce aktivacione funkcije
// Prvi parametar je obavezan, sve posle toga nije obavezno i ne vrsi se nikakva provera da li su ti dodatni parametri validni
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
