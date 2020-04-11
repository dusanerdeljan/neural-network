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
	namespace activation
	{
		enum Type
		{
			SIGMOID, RELU, LEAKY_RELU, ELU, TANH, SOFTMAX
		};

		class ActivationFunction
		{
		public:
			virtual Matrix Function(Matrix& x) = 0;
			virtual Matrix Derivative(Matrix& x) = 0;
			virtual Type GetType() const = 0;
			virtual void SaveActivationFunction(std::ofstream& out) const;
		};

		class Sigmoid : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class ReLu : public ActivationFunction
		{
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class LeakyReLu : public ActivationFunction
		{
		private:
			double alpha;
		public:
			LeakyReLu(double alpha = 0.1);
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class ELu : public ActivationFunction
		{
		private:
			double alpha;
		public:
			ELu(double alpha = 0.1);
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class Tanh : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};

		class Softmax : public ActivationFunction
		{
		private:
			Matrix m_Activation;
		public:
			Matrix Function(Matrix& x) override;
			Matrix Derivative(Matrix& x) override;
			Type GetType() const override;
		};
	}

	class ActivationFunctionFactory
	{
	public:
		static std::shared_ptr<activation::ActivationFunction> BuildActivationFunction(activation::Type type);
	};
}
