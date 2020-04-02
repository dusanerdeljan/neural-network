#pragma once
#include <random>
#include <memory>
#include "Matrix.h"

namespace nn
{
	namespace initialization
	{
		enum Type
		{
			RANDOM, XAVIER_UNIFORM, XAVIER_NORMAL, HE_UNIFORM, HE_NORMAL, LECUN_UNIFORM, LECUN_NORMAL, NONE
		};

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
			Random(double min = -1, double max = 1) : m_Min(min), m_Max(max) {}
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

	class WeightInitializerFactory
	{
	public:
		static std::shared_ptr<initialization::Initializer> BuildWeightInitializer(initialization::Type type)
		{
			switch (type)
			{
			case initialization::RANDOM:
				return std::make_shared<initialization::Random>();
			case initialization::XAVIER_UNIFORM:
				return std::make_shared<initialization::XavierUniform>();
			case initialization::XAVIER_NORMAL:
				return std::make_shared<initialization::XavierNormal>();
			case initialization::HE_UNIFORM:
				return std::make_shared<initialization::HeUniform>();
			case initialization::HE_NORMAL:
				return std::make_shared<initialization::HeNormal>();
			case initialization::LECUN_UNIFORM:
				return std::make_shared<initialization::LeCunUniform>();
			case initialization::LECUN_NORMAL:
				return std::make_shared<initialization::LeCunNormal>();
			default:
				return nullptr;
			}
		}
	};
}
