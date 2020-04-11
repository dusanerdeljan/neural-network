import sys
import os
import csv
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../PythonAPI/"))

"""
Example from:
https://www.freecodecamp.org/news/how-to-build-your-first-neural-network-to-predict-house-prices-with-keras-f8db83049159/
"""

from pynn.neuralnetwork import NeuralNetwork
from pynn.optimizers import Adam
from pynn.layers.dense import Dense

data = []
labels = []
max_row = np.zeros(shape=10, dtype=np.double)

with open('../dataset/housepricedata.csv') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    for row in reader:
        row = np.asarray(row, dtype=np.double)
        max_row = np.maximum(max_row, row[:-1])
        label = np.array([0, 1] if row[-1] > 0 else [1, 0], dtype=np.double)
        data.append(row[:-1])
        labels.append(label)
data = np.array(data) / max_row

train_length = int(0.8*len(data))
x_train = data[:train_length]
y_train = labels[:train_length]
x_test = data[train_length:]
y_test = labels[train_length:]

model = NeuralNetwork([
    Dense(500, 'relu', inputs=10),
    Dense(500, 'relu'),
    Dense(500, 'relu'),
    Dense(500, 'relu'),
    Dense(2, 'sigmoid')
])
model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error', initializer='xavier_normal', regularizer='l2')
model.fit(x_train, y_train, batch_size=32, epochs=100)

correct = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    if output.argmax == np.argmax(y):
        print("Correct")
        correct += 1
    else:
        print("Wrong")
print(f"Model accuracy: {correct}/{len(x_test)}")


