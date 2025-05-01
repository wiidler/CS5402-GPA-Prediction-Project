import numpy as np
import matplotlib.pyplot as plt
from Data_Parse import open_file

class SVM:
    def __init__(self, learning_rate=0.01, lamda_parameter=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.lamda = lamda_parameter
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # SVM training
        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    # Update weights only (regularization)
                    self.weights -= self.lr * (2 * self.lamda * self.weights)
                else:
                    # Update weights and bias
                    self.weights -= self.lr * (2 * self.lamda * self.weights - np.dot(x_i, y_[idx]))
                    self.bias += self.lr * y_[idx]

    def predict(self, X):
        # Return the natural signed distance (NOT thresholded)
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Load the data from the csv file
    X, y = open_file()

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of SVM
    model = SVM(learning_rate=0.01, lamda_parameter=0.01, num_iterations=1000)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make raw predictions (real values)
    predictions_raw = model.predict(X_test)

    # For evaluation: Threshold predictions at 0 to get 0 or 1
    predictions = np.where(predictions_raw >= 0, 1, 0)

    # Accuracy
    accuracy = np.mean(predictions == y_test)
    accuracy_text = f"Accuracy: {accuracy * 100:.2f}%"
    print(accuracy_text)

    # Confusion Matrix
    true_positive = np.sum((predictions == 1) & (y_test == 1))
    true_negative = np.sum((predictions == 0) & (y_test == 0))
    false_positive = np.sum((predictions == 1) & (y_test == 0))
    false_negative = np.sum((predictions == 0) & (y_test == 1))

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

    correct = predictions_sorted == y_sorted
    x_range = np.arange(len(y_sorted))

    plt.figure(figsize=(12, 6))
    plt.scatter(x_range, y_sorted, c=np.where(correct, 'green', 'red'), s=40, marker='o')
    plt.yticks([0, 1])
    plt.xlabel('Sample Index (sorted by Actual Label)')
    plt.ylabel('Actual Label')
    plt.title('SVM Predictions vs Actual Labels')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add textbox with accuracy and confusion matrix
    textbox_props = dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white', alpha=0.8)
    textbox_text = f"{accuracy_text}\n{confusion_matrix_text}"
    plt.gca().text(0.02, 0.95, textbox_text, transform=plt.gca().transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left', bbox=textbox_props)

    plt.tight_layout()
    plt.savefig('results/svm_results.png')
