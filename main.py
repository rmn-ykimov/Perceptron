import json

# import the Perceptron class
from perceptron import Perceptron

# create a perceptron model with a learning rate of 1 and 10 epochs
perceptron = Perceptron(lr=1, epochs=10)

# load training data from train.json
with open("train.json", "r") as f:
    data = json.load(f)
    X = data["X"]
    y = data["y"]

# fit the model on the training data
perceptron.fit(X, y)

# load test data from test.json
with open("data/test.json", "r") as f:
    data = json.load(f)
    X_test = data["X"]
    y_test = data["y"]

# make predictions on the test data
predictions = perceptron.predict(X_test)

# print the predictions
print(predictions)
