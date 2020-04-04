#pragma once
#include <memory>
#include "Matrix.h"

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
			double m_L1;
		public:
			L1Regularizer(double l1) : m_L1(l1) {}
			void Regularize(const Matrix& weights, Matrix& gradient) const override
			{
				gradient += Matrix::Map(weights, [l1 = m_L1](double x) { return x >= 0 ? l1 : -l1; });
			}
		};

		class L2Regularizer : public Regularizer
		{
		private:
			double m_L2;
		public:
			L2Regularizer(double l2) : m_L2(l2) {}
			void Regularize(const Matrix& weights, Matrix& gradient) const override
			{
				gradient += 2 * m_L2*weights;
			}
		};

		class L1L2Regularizer : public Regularizer
		{
		private:
			double m_L1;
			double m_L2;
		public:
			L1L2Regularizer(double l1, double l2) : m_L1(l1), m_L2(l2) {}
			void Regularize(const Matrix& weights, Matrix& gradient) const override
			{
				gradient += Matrix::Map(weights, [l1 = m_L1, l2 = m_L2](double x)
				{
					double sign = x >= 0 ? l1 : -l1;
					return sign * 2 * l2*x;
				});
			}
		};
	}

	class RegularizerFactory
	{
	public:
		static std::shared_ptr<regularizer::Regularizer> BuildRegularizer(regularizer::Type type, double l1 = 0.01, double l2 = 0.01)
		{
			switch (type)
			{
			case regularizer::NONE:
				return std::make_shared<regularizer::Regularizer>();
			case regularizer::L1:
				return std::make_shared<regularizer::L1Regularizer>(l1);
			case regularizer::L2:
				return std::make_shared<regularizer::L2Regularizer>(l2);
			case regularizer::L1L2:
				return std::make_shared<regularizer::L1L2Regularizer>(l1, l2);
			default:
				return nullptr;
			}
		}
	};
}
