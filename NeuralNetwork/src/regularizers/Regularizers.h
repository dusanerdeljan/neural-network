/*
Statically-linked deep learning library
Copyright (C) 2020 Dušan Erdeljan, Nedeljko Vignjević

This file is part of neural-network

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

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
