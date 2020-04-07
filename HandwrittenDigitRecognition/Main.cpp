#include "NeuralNetwork\NeuralNetwork.h"
#include <iostream>

typedef std::vector<nn::TrainingData> dataset;

std::pair<dataset, dataset> LoadData();
dataset LoadFromFile(const char* images, const char* labels, unsigned int capacity);
void Evaluate(nn::NeuralNetwork& model, const dataset& data);

// Switch to Release configuration
// Download MNIST dataset and change paths
static const char* TRAINING_IMAGES = "dataset/train-images.idx3-ubyte";
static const char* TRAINING_LABELS = "dataset/train-labels.idx1-ubyte";
static const char* TEST_IMAGES = "dataset/t10k-images.idx3-ubyte";
static const char* TEST_LABELS = "dataset/t10k-labels.idx1-ubyte";

int main()
{
	nn::NeuralNetwork model(784, {
		nn::Layer(784, 64, nn::activation::RELU),
		nn::Layer(64, 64, nn::activation::RELU),
		nn::Layer(64, 10, nn::activation::SOFTMAX)
	}, nn::initialization::XAVIER_NORMAL, nn::loss::CROSS_ENTROPY);
	auto data = LoadData();
	std::cout << "Dataset loaded." << std::endl;
	model.Train(nn::optimizer::Adam(0.001), 5, data.first, 32, nn::regularizer::L2);
	model.SaveModel("model.bin");
	Evaluate(model, data.second);
	std::cin.get();
	return 0;
}

void Evaluate(nn::NeuralNetwork& model, const dataset& data)
{
	int correct = 0;
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		auto prediction = model.Eval(data[i].inputs);
		unsigned int predictionValue = prediction.index;
		unsigned int maxIndex = std::max_element(data[i].target.begin(), data[i].target.end()) - data[i].target.begin();
		std::cout << "Prediction: " << predictionValue << ", True: " << maxIndex << " " << (predictionValue == maxIndex ? "CORRECT" : "WRONG") << std::endl;;
		if (predictionValue == maxIndex) correct++;
	}
	std::cout << "Correct: " << correct << " / " << data.size() << std::endl;
}

std::pair<dataset, dataset> LoadData()
{
	return{
		LoadFromFile(TRAINING_IMAGES, TRAINING_LABELS, 60000),
		LoadFromFile(TEST_IMAGES, TEST_LABELS, 10000)
	};
}

dataset LoadFromFile(const char * images, const char * labels, unsigned int capacity)
{
	std::ifstream imageFIle;
	imageFIle.open(images, std::ios::binary);
	imageFIle.seekg(16, std::ios::beg);
	std::ifstream labelsFile;
	labelsFile.open(labels, std::ios::binary);
	labelsFile.seekg(8, std::ios::beg);
	dataset trainData;
	for (unsigned int i = 0; i < capacity; ++i)
	{
		unsigned char* image = new unsigned char[784];
		imageFIle.read(reinterpret_cast<char*>(image), sizeof(unsigned char) * 784);
		unsigned char label;
		labelsFile.read((char*)&label, sizeof(label));
		std::vector<double> data;
		std::for_each(image, image + 784, [&data](double x) { data.push_back(x / 255.0); });
		std::vector<double> result(10);
		std::fill(result.begin(), result.end(), 0);
		result[label] = 1;
		trainData.push_back({ data , result });
		delete[] image;
	}
	return trainData;
}