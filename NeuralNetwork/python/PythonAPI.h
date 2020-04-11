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
#include "../NeuralNetwork.h"

#define NN_API extern "C" __declspec(dllexport)

typedef struct sgd
{
	double lr;
} SGD;

typedef struct momentum
{
	double lr;
	double moment;
} Momentum;

typedef struct nesterov
{
	double lr;
	double moment;
} Nesterov;

typedef struct adagrad
{
	double lr;
} Adagrad;

typedef struct rmsprop
{
	double lr;
	double beta;
} RMSProp;

typedef struct adadelta
{
	double lr;
	double beta;
} Adadelta;

typedef struct adam
{
	double lr;
	double beta1;
	double beta2;
} Adam;

typedef struct nadam
{
	double lr;
	double beta1;
	double beta2;
} Nadam;

typedef struct adamax
{
	double lr;
	double beta1;
	double beta2;
} Adamax;

typedef struct amsgrad
{
	double lr;
	double beta1;
	double beta2;
} AMSGrad;

typedef struct output
{
	double value;
	unsigned int argmax;
} Output;

typedef struct dense
{
	unsigned int neurons;
	unsigned int activation_function;
	unsigned int inputs;
} Dense;

typedef struct model
{
	std::unique_ptr<nn::NeuralNetwork> net;
	std::unique_ptr<nn::optimizer::Optimizer> optimizer;
	std::vector<nn::Layer> layers;
	unsigned int inputSize;
	unsigned int outputSize;
	unsigned int regularizerType;
} Model;

static Model model;
static std::vector<nn::TrainingData> trainingData;
static const double default_lr = 0.01;

void create_optimizer(nn::optimizer::Type type, void* ptr = NULL)
{
	switch (type)
	{
	case nn::optimizer::Type::GRADIENT_DESCENT:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::GradientDescent>(default_lr);
		else
		{
			SGD* sgd = (SGD*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::GradientDescent>(sgd->lr);
		}
		break;
	case nn::optimizer::Type::MOMENTUM:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Momentum>(default_lr);
		else
		{
			Momentum* momentum = (Momentum*)ptr;
			model.optimizer = std::make_unique <nn::optimizer::Momentum>(momentum->lr, momentum->moment);
		}
		break;
	case nn::optimizer::Type::NESTEROV:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Nesterov>(default_lr);
		else
		{
			Nesterov* nesterov = (Nesterov*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Nesterov>(nesterov->lr, nesterov->moment);
		}
		break;
	case nn::optimizer::Type::ADAGRAD:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Adagrad>(default_lr);
		else
		{
			Adagrad* adagrad = (Adagrad*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Adagrad>(adagrad->lr);
		}
		break;
	case nn::optimizer::Type::RMSPROP:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::RMSProp>(default_lr);
		else
		{
			RMSProp* rmsprop = (RMSProp*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::RMSProp>(rmsprop->lr, rmsprop->beta);
		}
		break;
	case nn::optimizer::Type::ADADELTA:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Adadelta>(default_lr);
		else
		{
			Adadelta* adadelta = (Adadelta*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Adadelta>(adadelta->lr, adadelta->beta);
		}
		break;
	case nn::optimizer::Type::ADAM:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Adam>(default_lr);
		else
		{
			Adam* adam = (Adam*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Adam>(adam->lr, adam->beta1, adam->beta2);
		}
		break;
	case nn::optimizer::Type::NADAM:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Nadam>(default_lr);
		else
		{
			Nadam* nadam = (Nadam*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Nadam>(nadam->lr, nadam->beta1, nadam->beta2);
		}
		break;
	case nn::optimizer::Type::ADAMAX:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::Adamax>(default_lr);
		else
		{
			Adamax* adamax = (Adamax*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::Adamax>(adamax->lr, adamax->beta1, adamax->beta2);
		}
		break;
	case nn::optimizer::Type::AMSGRAD:
		if (ptr == NULL)
			model.optimizer = std::make_unique<nn::optimizer::AMSGrad>(default_lr);
		else
		{
			AMSGrad* amsgrad = (AMSGrad*)ptr;
			model.optimizer = std::make_unique<nn::optimizer::AMSGrad>(amsgrad->lr, amsgrad->beta1, amsgrad->beta2);
		}
		break;
	}
}


NN_API void compile(unsigned int optimizer, unsigned int loss, unsigned int initializer, unsigned int regularizer)
{
	create_optimizer(nn::optimizer::Type(optimizer));
	model.regularizerType = regularizer;
	model.net = std::make_unique<nn::NeuralNetwork>(nn::NeuralNetwork(model.inputSize, std::move(model.layers), nn::initialization::Type(initializer), nn::loss::Type(loss)));
}

NN_API void compile_optimizer(void* ptrOptimizer, unsigned int optimizer, unsigned int loss, unsigned int initializer, unsigned int regularizer)
{
	create_optimizer(nn::optimizer::Type(optimizer), ptrOptimizer);
	compile(optimizer, loss, initializer, regularizer);
}

NN_API void add(Dense* dense)
{
	if (model.layers.empty()) model.inputSize = dense->inputs;
	model.layers.emplace_back(dense->inputs, dense->neurons, nn::activation::Type(dense->activation_function));
	model.outputSize = dense->neurons;
}

NN_API void save(const char* file_name)
{
	model.net->SaveModel(file_name);
}

NN_API void load(const char* file_name)
{
	model.net = std::make_unique<nn::NeuralNetwork>(nn::NeuralNetwork::LoadModel(file_name));
}

NN_API void state_loaded(void* ptrOptimizer, unsigned int optimizer, unsigned int regularizer, unsigned int inputSize, unsigned int outputSize)
{
	create_optimizer(nn::optimizer::Type(optimizer), ptrOptimizer);
	model.inputSize = inputSize;
	model.outputSize = outputSize;
	model.regularizerType = regularizer;
}

NN_API void add_training_sample(double inputs[], double targets[])
{
	trainingData.emplace_back(std::vector<double>(inputs, inputs + model.inputSize), std::vector<double>(targets, targets + model.outputSize));
}

NN_API void train(unsigned int epochs, unsigned int batchSize)
{
	model.net->Train(*(model.optimizer), epochs, trainingData, batchSize, nn::regularizer::Type(model.regularizerType));
	trainingData.clear();
}

NN_API Output eval(double inputs[])
{
	nn::Output out = model.net->Eval(std::vector<double>(inputs, inputs + model.inputSize));
	Output o; o.value = out.value; o.argmax = out.index;
	return o;
}