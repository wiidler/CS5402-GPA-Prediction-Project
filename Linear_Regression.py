import numpy as np
import matplotlib.pyplot as plt
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
    # Load the data
    X, y = open_file()

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(learning_rate=0.001, num_iterations=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Threshold predictions for evaluation only
    predicted_labels = np.where(predictions >= 0.5, 1, 0)

    # Accuracy
    accuracy = np.mean(predicted_labels == y_test)
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}% (Threshold=0.5)"
    print(accuracy_text)

    # Confusion Matrix
    true_positive = np.sum((predicted_labels == 1) & (y_test == 1))
    true_negative = np.sum((predicted_labels == 0) & (y_test == 0))
    false_positive = np.sum((predicted_labels == 1) & (y_test == 0))
    false_negative = np.sum((predicted_labels == 0) & (y_test == 1))

    confusion_matrix_text = (
        f"TP (Correct Pass): {true_positive}\n"
        f"TN (Correct Fail): {true_negative}\n"
        f"FP (Incorrect Pass): {false_positive}\n"
        f"FN (Incorrect Fail): {false_negative}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix_text)

    # Plot sorted
    sorted_indices = np.argsort(y_test)
    y_sorted = y_test[sorted_indices]
    predictions_sorted = predictions[sorted_indices]
    predicted_labels_sorted = predicted_labels[sorted_indices]

    correct = predicted_labels_sorted == y_sorted
    x_range = np.arange(len(y_sorted))

    plt.figure(figsize=(12, 6))
    plt.plot(x_range, predictions_sorted, label='Predicted Value (Continuous)', color='blue', linewidth=2)
    plt.scatter(x_range, y_sorted, c=np.where(correct, 'green', 'red'), s=40, marker='o', label='Actual Label')
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Pass/Fail Threshold (0.5)')
    plt.yticks([0, 0.5, 1])
    plt.xlabel('Sample Index (sorted by Actual Label)')
    plt.ylabel('Prediction / Actual')
    plt.title('Linear Regression Predictions vs Actual Labels')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Add textbox with accuracy and confusion matrix
    textbox_props = dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white', alpha=0.8)
    textbox_text = f"{accuracy_text}\n{confusion_matrix_text}"
    plt.gca().text(0.02, 0.95, textbox_text, transform=plt.gca().transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left', bbox=textbox_props)

    plt.tight_layout()
    plt.savefig('linear_regression_results.png')
