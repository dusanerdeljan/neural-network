#pragma once
#include <random>
#include "Matrix.h"

namespace Initialization
{
	class Initializer
	{
	public:
		virtual void Initialize(Matrix& matrix) const = 0;
	};

	class Random : public Initializer
	{
	private:
		double m_Min;
		double m_Max;
	public:
		Random(double min, double max) : m_Min(min), m_Max(max) {}
		void Initialize(Matrix& matrix) const override
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(m_Min, m_Max);
			matrix.Map([&valueDistribution, &engine](double x)
			{
				return valueDistribution(engine);
			});
		}
	};

	class XavierUniform : public Initializer
	{
		void Initialize(Matrix& matrix) const override
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(6.0 / (matrix.GetWidth() + matrix.GetHeight()));
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return (valueDistribution(engine) - 0.5) * factor;
			});
		}
	};

	class XavierNormal : public Initializer
	{
		void Initialize(Matrix& matrix) const override
		{
			std::default_random_engine engine;
			std::normal_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(6.0 / (matrix.GetWidth() + matrix.GetHeight()));
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return valueDistribution(engine) * factor;
			});
		}
	};

	class LeCunUniform : public Initializer
	{
		void Initialize(Matrix& matrix) const override
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(3.0 / matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return (valueDistribution(engine) - 0.5) * factor;
			});
		}
	};

	class LeCunNormal : public Initializer
	{
		void Initialize(Matrix& matrix) const override
		{
			std::default_random_engine engine;
			std::normal_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 1.0 / sqrt(matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return valueDistribution(engine) * factor;
			});
		}
	};

	class HeUniform : public Initializer
	{
		void Initialize(Matrix& matrix) const override
		{
			std::random_device randomDevice;
			std::mt19937 engine(randomDevice());
			std::uniform_real_distribution<double> valueDistribution(0.0, 1.0);
			double factor = 2.0 * sqrt(6.0 / matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return (valueDistribution(engine) - 0.5) * factor;
			});
		}
	};

	class HeNormal : public Initializer
	{
		void Initialize(Matrix& matrix) const override
		{
			std::default_random_engine engine;
			std::normal_distribution<double> valueDistribution(0.0, 1.0);
			double factor = sqrt(2.0 / matrix.GetWidth());
			matrix.Map([factor, &valueDistribution, &engine](double x)
			{
				return valueDistribution(engine) * factor;
			});
		}
	};
}
