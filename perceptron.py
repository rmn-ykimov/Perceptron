class Perceptron:
    """
    A simple Perceptron implementation that can be trained to classify linearly
     separable data.
    """
    def __init__(self, lr=1, epochs=10):
        """
        lr: float, learning rate.
        epochs: int, number of iterations to train the model.
        """
        self.weights = []
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X, y):
        """
        X: List of feature vectors, each feature vector is a list of floats.
        y: List of labels, each label is an int.
        """
        self.weights = [0 for _ in range(len(X[0]) + 1)]
        for _ in range(self.epochs):
            for i in range(len(X)):
                x, true_y = X[i], y[i]
                x.append(1) # add bias term
                predicted_y = 1 if sum(w*f for w,f in zip(self.weights, x)) >= 0 else -1
                error = true_y - predicted_y
                self.weights = [w + self.lr*error*f for w,f in zip(self.weights, x)]
                x.pop() # remove bias term
    
    def predict(self, X):
        """
        X: List of feature vectors, each feature vector is a list of floats.
        returns: List of predicted labels, each label is an int.
        """
        predictions = []
        for x in X:
            x.append(1) # add bias term
            predictions.append(1 if sum(w*f for w,f in zip(self.weights, x)) >= 0 else -1)
            x.pop() # remove bias term
        return predictions
