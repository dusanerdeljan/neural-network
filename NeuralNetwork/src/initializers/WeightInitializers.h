#pragma once
#include <random>
#include <memory>
#include "../math/Matrix.h"

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
			double m_Min = -1;
			double m_Max = 1;
		public:
			void Initialize(Matrix& matrix) const override;
		};

		class XavierUniform : public Initializer
		{
			void Initialize(Matrix& matrix) const override;
		};

		class XavierNormal : public Initializer
		{
			void Initialize(Matrix& matrix) const override;
		};

		class LeCunUniform : public Initializer
		{
			void Initialize(Matrix& matrix) const override;
		};

		class LeCunNormal : public Initializer
		{
			void Initialize(Matrix& matrix) const override;
		};

		class HeUniform : public Initializer
		{
			void Initialize(Matrix& matrix) const override;
		};

		class HeNormal : public Initializer
		{
			void Initialize(Matrix& matrix) const override;
		};
	}

	class WeightInitializerFactory
	{
	public:
		static std::shared_ptr<initialization::Initializer> BuildWeightInitializer(initialization::Type type);
	};
}
