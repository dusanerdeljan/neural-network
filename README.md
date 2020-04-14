# neural-network
Statically-linked deep learning library written in C++ from scratch. Library also offers python interface.

## Features

Currently supported featuers:

 * **Optimizers**: SGD, SGD with Momentum, SGD with Nesterov Momentum, Adagrad, Adam, Nadam, Adamax, AMSGrad, Adadelta, RMSProp, Adabound, AMSBound
 * **Activation functions**: Sigmoid, ReLU, Leaky ReLU, ELU, Tanh, Softmax
 * **Loss functions**: Mean Absolute Error, Mean Squared Error, Quadratic, Half Quadratic, Cross Entropy, NLL
 * **Weight initializers**: Random, Xavier Uniform, Xavier Normal, LeCun Uniform, LeCun Normal, He Uniform, He Normal
 * **Regularizers**: L1, L2, L1L2, None
 * **Layers**: Dense (Fully connected)
 
## Example usage

### C++

#### Creating a model

```cpp
nn::NeuralNetwork model(784, {
	nn::Layer(784, 64, nn::activation::RELU),
	nn::Layer(64, 64, nn::activation::RELU),
	nn::Layer(64, 10, nn::activation::SOFTMAX)
}, nn::initialization::LECUN_UNIFORM, nn::loss::NLL);
```

#### Training a model

```cpp
std::vector<nn::TrainingData> trainingData = GetTrainingData();
unsigned int epochs = 10;
unsigned int batchSize = 10;
model.Train(nn::optimizer::Adam(0.001), epochs, trainingData, batchSize, nn::regularizer::L2);
```

#### Evaluating a model

```cpp
std::vector<double> input = LoadImage();
nn::Output output = model.Eval(input);
std::cout << "Predicted class: " << output.Argmax << " (" << output.Value*100 << "%)" << std::endl;
```

#### Saving and loading a model

```cpp
model.SaveModel("model.bin");
nn::NeuralNetwork model2 = nn::NeuralNetwork::LoadModel("model.bin");
```

### Python

#### Creating a model

##### Initializing a model in the constructor

```python
model = NeuralNetwork([
    Dense(64, 'relu', inputs=784),
    Dense(64, 'relu'),
    Dense(10, 'softmax')
])
```
##### Adding layers manually

```python
model = NeuralNetwork()
model.add(Dense(64, 'relu', inputs=784))
model.add(Dense(64, 'relu'))
model.add(Dense(10, 'softmax'))
```

#### Compiling a model

##### Using predefined optimizer options

```python
model.compile(optimizer='adam', loss='nll', initializer='lecun_uniform', regularizer='l2')
```
##### Using custom optimizer options

```python
model.compile(optimizer=Adam(lr=0.001, beta1=0.9, beta2=0.999), loss='nll', initializer='lecun_uniform', regularizer='l2')
```

#### Training a model

```python
# x and y can be python lists or numpy arrays
x, y = get_training_data()
model.fit(x, y, epochs=10, batch_size=10)
```

#### Evaluating a model

```python
# inputs can be python list or numpy array
inputs = load_image()
output = model.predict(inputs)
print(f"Predicted class: {output.argmax} ( {output.value*100}% )")
```

#### Saving and loading a model

```python
model.save('model.bin')
model2 = NeuralNetwork.load('model.bin')
```

## License

This program is free.</br>
You can redistribute it and/or change it under the terms of **GNU General Public License version 3.0** (GPLv3). </br>
You can find a copy of the license in the repository.
