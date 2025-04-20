import numpy as np
from Data_Parse import open_file

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1.0 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1.0 / num_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
if __name__ == "__main__":
    # Load the data from the csv file
    data_array = open_file()
    # Split the data into features and target variable
    X = data_array[:, :-1]  # All columns except the last one
    y = data_array[:, -1]   # Last column

    # Create an instance of LinearRegression
    model = LinearRegression(learning_rate=0.001, num_iterations=1000)

    # Fit the model to the data
    model.fit(X, y)

    # Make predictions on the training set
    predictions = model.predict(X)

    # Print the first 5 predictions
    print("First 5 predictions:", predictions[:5])