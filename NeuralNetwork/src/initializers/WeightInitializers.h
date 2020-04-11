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
