#pragma once
#include <memory>
#include "../math/Matrix.h"

namespace nn
{
	namespace regularizer
	{
		enum Type
		{
			NONE, L1, L2, L1L2
		};

		class Regularizer
		{
		public:
			virtual void Regularize(const Matrix& weights, Matrix& gradient) const {}
		};

		class L1Regularizer : public Regularizer
		{
		private:
			double m_L1 = 0.01;
		public:
			void Regularize(const Matrix& weights, Matrix& gradient) const override;
		};

		class L2Regularizer : public Regularizer
		{
		private:
			double m_L2 = 0.01;
		public:
			void Regularize(const Matrix& weights, Matrix& gradient) const override;
		};

		class L1L2Regularizer : public Regularizer
		{
		private:
			double m_L1 = 0.01;
			double m_L2 = 0.01;
		public:
			void Regularize(const Matrix& weights, Matrix& gradient) const override;
		};
	}

	class RegularizerFactory
	{
	public:
		static std::shared_ptr<regularizer::Regularizer> BuildRegularizer(regularizer::Type type);
	};
}
