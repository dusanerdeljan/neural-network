#pragma once
#include <cmath>
#include <unordered_map>

namespace nn
{
	namespace optimizer
	{
		enum class Type
		{
			SGD, MOMENTUM, NESTEROV, ADAGRAD, RMSPROP, ADADELTA, ADAM, NADAM
		};

		class Optimizer
		{
		protected:
			double m_LearningRate;
			Optimizer(double lr) : m_LearningRate(lr) {}
		public:
			virtual void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) = 0;
			virtual void Reset() {}
		};

		class SGD : public Optimizer
		{
		public:
			SGD(double lr) : Optimizer(lr) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				layer.m_WeightMatrix -= m_LearningRate * gradient * previousActivation.Transpose();
				layer.m_BiasMatrix -= m_LearningRate * gradient;
			}
		};

		class Momentum : public Optimizer
		{
		private:
			double m_Momentum;
			std::unordered_map<unsigned int, Matrix> lastDeltaWeight;
			std::unordered_map<unsigned int, Matrix> lastDeltaBias;
		public:
			Momentum(double lr, double momentum = 0.9) : Optimizer(lr), m_Momentum(momentum) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;
				if (lastDeltaWeight.find(layerIndex) == lastDeltaWeight.end())
				{
					lastDeltaWeight[layerIndex] = deltaWeight*m_LearningRate;
					lastDeltaBias[layerIndex] = deltaBias*m_LearningRate;
				}
				else
				{
					lastDeltaWeight[layerIndex] *= m_Momentum;
					lastDeltaWeight[layerIndex] += deltaWeight * m_LearningRate;
					lastDeltaBias[layerIndex] *= m_Momentum;
					lastDeltaBias[layerIndex] += deltaBias * m_LearningRate;
				}

				layer.m_WeightMatrix -= lastDeltaWeight[layerIndex];
				layer.m_BiasMatrix -= lastDeltaBias[layerIndex];
			}
			void Reset() override
			{
				lastDeltaWeight.clear();
				lastDeltaBias.clear();
			}
		};

		class Nesterov : public Optimizer
		{
		private:
			double m_Momentum;
			std::unordered_map<unsigned int, Matrix> lastDeltaWeight;
			std::unordered_map<unsigned int, Matrix> lastDeltaBias;
		public:
			Nesterov(double lr, double momentum = 0.9) : Optimizer(lr), m_Momentum(momentum) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;
				if (lastDeltaWeight.find(layerIndex) == lastDeltaWeight.end())
				{
					lastDeltaWeight[layerIndex] = Matrix::Map(deltaWeight, [](double x) { return 0.0; });
					lastDeltaBias[layerIndex] = Matrix::Map(deltaBias, [](double x) { return 0.0; });
				}
				Matrix tempWeight = lastDeltaWeight[layerIndex];
				lastDeltaWeight[layerIndex] *= m_Momentum;
				lastDeltaWeight[layerIndex] -= deltaWeight * m_LearningRate;
				Matrix tempBias = lastDeltaBias[layerIndex];
				lastDeltaBias[layerIndex] *= m_Momentum;
				lastDeltaBias[layerIndex] -= deltaBias*m_LearningRate;
				layer.m_WeightMatrix += (tempWeight*(-m_Momentum)) + (lastDeltaWeight[layerIndex] * (1 + m_Momentum));
				layer.m_BiasMatrix += (tempBias*(-m_Momentum)) + (lastDeltaBias[layerIndex] * (1 + m_Momentum));
			}
			void Reset() override
			{
				lastDeltaWeight.clear();
				lastDeltaBias.clear();
			}
		};

		class Adagrad : public Optimizer
		{
		private:
			std::unordered_map<unsigned int, Matrix> gradSquaredW;
			std::unordered_map<unsigned int, Matrix> gradSquaredB;
		public:
			Adagrad(double lr) : Optimizer(lr) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;

				if (gradSquaredW.find(layerIndex) == gradSquaredW.end())
				{
					gradSquaredW[layerIndex] = Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[layerIndex] = Matrix::Map(deltaBias, [](double x) { return x*x; });
				}
				else
				{
					gradSquaredW[layerIndex] += Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[layerIndex] += Matrix::Map(deltaBias, [](double x) { return x*x; });
				}

				Matrix deljenikW = Matrix::Map(gradSquaredW[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
				Matrix deljenikB = Matrix::Map(gradSquaredB[layerIndex], [](double x) { return sqrt(x) + 1e-7; });

				layer.m_WeightMatrix -= (m_LearningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
				layer.m_BiasMatrix -= (m_LearningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));;
			}
			void Reset() override
			{
				gradSquaredB.clear();
				gradSquaredW.clear();
			}
		};

		class RMSProp : public Optimizer
		{
		private:
			double m_Beta;
			std::unordered_map<unsigned int, Matrix> gradSquaredW;
			std::unordered_map<unsigned int, Matrix> gradSquaredB;
		public:
			RMSProp(double lr, double beta = 0.99) : Optimizer(lr), m_Beta(beta) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;
				if (gradSquaredW.find(layerIndex) == gradSquaredW.end())
				{
					gradSquaredW[layerIndex] = (1 - m_Beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[layerIndex] = (1 - m_Beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				}
				else
				{
					gradSquaredW[layerIndex] = m_Beta * gradSquaredW[layerIndex] + (1 - m_Beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[layerIndex] = m_Beta * gradSquaredB[layerIndex] + (1 - m_Beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
				}
				Matrix deljenikW = Matrix::Map(gradSquaredW[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
				Matrix deljenikB = Matrix::Map(gradSquaredB[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
				layer.m_WeightMatrix -= (m_LearningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
				layer.m_BiasMatrix -= (m_LearningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));
			}
			void Reset() override
			{
				gradSquaredB.clear();
				gradSquaredW.clear();
			}
		};

		class Adadelta : public Optimizer
		{
		private:
			double m_Beta;
			std::unordered_map<unsigned int, Matrix> gradSquaredW;
			std::unordered_map<unsigned int, Matrix> gradSquaredB;
			//std::unordered_map<unsigned int, Matrix> deW;
			//std::unordered_map<unsigned int, Matrix> deB;
		public:
			Adadelta(double lr, double beta = 0.99) : Optimizer(lr), m_Beta(beta) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;
				if (gradSquaredW.find(layerIndex) == gradSquaredW.end())
				{
					gradSquaredW[layerIndex] = (1 - m_Beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[layerIndex] = (1 - m_Beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					//deW[layerIndex] = (1 - beta) * Matrix::Map(m_Layers[layerIndex].m_WeightMatrix, [](double x) { return x*x; });
					//deB[layerIndex] = (1 - beta) * Matrix::Map(m_Layers[layerIndex].m_BiasMatrix, [](double x) { return x*x; });
				}
				else
				{
					gradSquaredW[layerIndex] = m_Beta * gradSquaredW[layerIndex] + (1 - m_Beta) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					gradSquaredB[layerIndex] = m_Beta * gradSquaredB[layerIndex] + (1 - m_Beta) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					//Matrix deltaW = (m_Layers[layerIndex].m_WeightMatrix - deltaWeight);
					//Matrix deltaB = (m_Layers[layerIndex].m_BiasMatrix - deltaBias);
					//deW[layerIndex] = beta * deW[layerIndex] + (1 - beta) * Matrix::Map(deltaW, [](double x) { return x*x; });
					//deB[layerIndex] = beta * deB[layerIndex] + (1 - beta) * Matrix::Map(deltaB, [](double x) { return x*x; });
				}
				Matrix deljenikW = Matrix::Map(gradSquaredW[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
				Matrix deljenikB = Matrix::Map(gradSquaredB[layerIndex], [](double x) { return sqrt(x) + 1e-7; });
				//Matrix learningRateW = Matrix::Map(deW[layerIndex], [](double x) { return sqrt(x); }) + Matrix(deW[layerIndex].GetHeight(), deW[layerIndex].GetWidth(), 1e-7);
				//Matrix learningRateB = Matrix::Map(deB[layerIndex], [](double x) { return sqrt(x); }) + Matrix(deB[layerIndex].GetHeight(), deB[layerIndex].GetWidth(), 1e-7);
				layer.m_WeightMatrix -= (m_LearningRate * deltaWeight).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));
				layer.m_BiasMatrix -= (m_LearningRate * deltaBias).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));
			}
			void Reset() override
			{
				gradSquaredB.clear();
				gradSquaredW.clear();
			}
		};

		class Adam : public Optimizer
		{
		private:
			double m_Beta1;
			double m_Beta2;
			// Weights
			std::unordered_map<unsigned int, Matrix> firstMomentW;
			std::unordered_map<unsigned int, Matrix> secondMomentW;
			// Biases
			std::unordered_map<unsigned int, Matrix> firstMomentB;
			std::unordered_map<unsigned int, Matrix> secondMomentB;
		public:
			Adam(double lr, double beta1 = 0.9, double beta2 = 0.999) : Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;

				Matrix firstUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());
				Matrix secondUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());

				Matrix firstUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());
				Matrix secondUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());

				if (firstMomentW.find(layerIndex) == firstMomentW.end())
				{
					// Weights
					firstMomentW[layerIndex] = (1 - m_Beta1) * deltaWeight;
					secondMomentW[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));

					// Biases
					firstMomentB[layerIndex] = (1 - m_Beta1) * deltaBias;
					secondMomentB[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));
				}
				else
				{
					// Weights
					firstMomentW[layerIndex] = firstMomentW[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaWeight;
					secondMomentW[layerIndex] = secondMomentW[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));

					// Biases
					firstMomentB[layerIndex] = firstMomentB[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaBias;
					secondMomentB[layerIndex] = secondMomentB[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));
				}

				Matrix deljenikW = Matrix::Map(secondUnbiasW, [](double x) { return sqrt(x) + 1e-7; });
				Matrix weight = (m_LearningRate * firstUnbiasW).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));

				Matrix deljenikB = Matrix::Map(secondUnbiasB, [](double x) { return sqrt(x) + 1e-7; });
				Matrix bias = (m_LearningRate * firstUnbiasB).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));

				layer.m_WeightMatrix -= weight;
				layer.m_BiasMatrix -= bias;
			}
			void Reset() override
			{
				firstMomentW.clear();
				firstMomentB.clear();
				secondMomentW.clear();
				secondMomentB.clear();
			}
		};

		class Nadam : public Optimizer
		{
		private:
			double m_Beta1;
			double m_Beta2;
			// Weights
			std::unordered_map<unsigned int, Matrix> firstMomentW;
			std::unordered_map<unsigned int, Matrix> secondMomentW;
			// Biases
			std::unordered_map<unsigned int, Matrix> firstMomentB;
			std::unordered_map<unsigned int, Matrix> secondMomentB;
		public:
			Nadam(double lr, double beta1 = 0.9, double beta2 = 0.999) : Optimizer(lr), m_Beta1(beta1), m_Beta2(beta2) {}
			void UpdateLayer(Layer& layer, Matrix& gradient, Matrix& previousActivation, int layerIndex = 0, unsigned int epoch = 0) override
			{
				Matrix deltaWeight = gradient * previousActivation.Transpose();
				Matrix deltaBias = gradient;

				Matrix firstUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());
				Matrix secondUnbiasW(deltaWeight.GetWidth(), deltaWeight.GetHeight());

				Matrix firstUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());
				Matrix secondUnbiasB(deltaBias.GetWidth(), deltaBias.GetHeight());

				if (firstMomentW.find(layerIndex) == firstMomentW.end())
				{
					// Weights
					firstMomentW[layerIndex] = (1 - m_Beta1) * deltaWeight;
					secondMomentW[layerIndex] = (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));

					// Biases
					firstMomentB[layerIndex] = (1 - m_Beta1) * deltaBias;
					secondMomentB[layerIndex] = Matrix::Map(deltaBias, [](double x) { return abs(x); });
					firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));
				}
				else
				{
					// Weights
					firstMomentW[layerIndex] = firstMomentW[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaWeight;
					secondMomentW[layerIndex] = secondMomentW[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaWeight, [](double x) { return x*x; });
					firstUnbiasW = firstMomentW[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasW = secondMomentW[layerIndex] / (1 - pow(m_Beta2, epoch));

					// Biases
					firstMomentB[layerIndex] = firstMomentB[layerIndex] * m_Beta1 + (1 - m_Beta1) * deltaBias;
					secondMomentB[layerIndex] = secondMomentB[layerIndex] * m_Beta2 + (1 - m_Beta2) * Matrix::Map(deltaBias, [](double x) { return x*x; });
					firstUnbiasB = firstMomentB[layerIndex] / (1 - pow(m_Beta1, epoch));
					secondUnbiasB = secondMomentB[layerIndex] / (1 - pow(m_Beta2, epoch));
				}

				Matrix deljenikW = Matrix::Map(secondUnbiasW, [](double x) { return sqrt(x) + 1e-7; });
				Matrix weight = (m_LearningRate * (firstUnbiasW * m_Beta1 + (1 - m_Beta1) / (1 - pow(m_Beta1, epoch)) * deltaWeight)).DotProduct(Matrix::Map(deljenikW, [](double x) { return 1 / x; }));

				Matrix deljenikB = Matrix::Map(secondUnbiasB, [](double x) { return sqrt(x) + 1e-7; });
				Matrix bias = (m_LearningRate * (firstUnbiasB * m_Beta1 + (1 - m_Beta1) / (1 - pow(m_Beta1, epoch)) * deltaBias)).DotProduct(Matrix::Map(deljenikB, [](double x) { return 1 / x; }));

				layer.m_WeightMatrix -= weight;
				layer.m_BiasMatrix -= bias;
			}
			void Reset() override
			{
				firstMomentW.clear();
				firstMomentB.clear();
				secondMomentW.clear();
				secondMomentB.clear();
			}
		};
	}
}