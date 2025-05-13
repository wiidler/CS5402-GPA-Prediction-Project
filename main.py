import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from Data_Parse import open_file
from Linear_Regression import LinearRegression
from Logistic_Regression import LogisticRegression
from SVM import SVM
from sklearn.metrics import confusion_matrix
import seaborn as sns

if __name__ == "__main__":
    # === Load and Split Data ===
    X, y = open_file()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === Train Models ===
    linear_model = LinearRegression(learning_rate=0.001, num_iterations=1000)
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)
    linear_predicted_labels = np.where(linear_predictions >= 0.5, 1, 0)

    logistic_model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    logistic_model.fit(X_train, y_train)
    logistic_probs = logistic_model.predict_proba(X_test)
    logistic_predicted_labels = logistic_model.predict(X_test)

    svm_model = SVM(learning_rate=0.01, lamda_parameter=0.01, num_iterations=1000)
    svm_model.fit(X_train, y_train)
    svm_scores = svm_model.predict(X_test)
    svm_predicted_labels = np.where(svm_scores >= 0, 1, 0)

    # === ROC Curve Computation ===
    fpr = {}
    tpr = {}
    roc_auc = {}

    fpr["Linear"], tpr["Linear"], _ = roc_curve(y_test, linear_predictions)
    fpr["Logistic"], tpr["Logistic"], _ = roc_curve(y_test, logistic_probs)
    fpr["SVM"], tpr["SVM"], _ = roc_curve(y_test, svm_scores)

    roc_auc["Linear"] = auc(fpr["Linear"], tpr["Linear"])
    roc_auc["Logistic"] = auc(fpr["Logistic"], tpr["Logistic"])
    roc_auc["SVM"] = auc(fpr["SVM"], tpr["SVM"])

    # === Plot ROC Curves ===
    plt.figure(figsize=(10, 8))
    for model in ["Linear", "Logistic", "SVM"]:
        plt.plot(fpr[model], tpr[model], label=f"{model} (AUC = {roc_auc[model]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/roc_comparison.png")

    # === Sort Data for Plotting ===
    sorted_indices = np.argsort(y_test)
    y_sorted = y_test[sorted_indices]

    linear_predictions_sorted = linear_predictions[sorted_indices]
    linear_labels_sorted = linear_predicted_labels[sorted_indices]

    logistic_probs_sorted = logistic_probs[sorted_indices]
    logistic_labels_sorted = logistic_predicted_labels[sorted_indices]

    svm_scores_sorted = svm_scores[sorted_indices]
    svm_labels_sorted = svm_predicted_labels[sorted_indices]

    x_range = np.arange(len(y_sorted))

    # === Plotting ===
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    models_info = [
        (
            axs[0],
            linear_predictions_sorted,
            linear_labels_sorted,
            "Linear Regression",
            True,
        ),
        (
            axs[1],
            logistic_probs_sorted,
            logistic_labels_sorted,
            "Logistic Regression",
            True,
        ),
        (axs[2], svm_labels_sorted, svm_labels_sorted, "SVM (Pass/Fail)", False),
    ]

    for ax, model_output, model_labels, title, is_continuous in models_info:
        correct = model_labels == y_sorted

        if is_continuous:
            ax.plot(
                x_range,
                model_output,
                color="blue",
                label="Predicted (Continuous)",
                linewidth=2,
            )
            ax.axhline(
                0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)"
            )
        else:
            ax.scatter(
                x_range,
                model_output,
                c=np.where(correct, "green", "red"),
                s=40,
                marker="x",
                label="Predicted Label",
            )

        ax.scatter(
            x_range,
            y_sorted,
            c=np.where(correct, "green", "red"),
            s=40,
            marker="o",
            label="Actual Label",
        )
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Calculate Accuracy and Confusion Matrix
        accuracy = np.mean(model_labels == y_sorted)
        true_positive = np.sum((model_labels == 1) & (y_sorted == 1))
        true_negative = np.sum((model_labels == 0) & (y_sorted == 0))
        false_positive = np.sum((model_labels == 1) & (y_sorted == 0))
        false_negative = np.sum((model_labels == 0) & (y_sorted == 1))

        textbox_text = (
            f"Accuracy: {accuracy * 100:.2f}%\n"
            f"TP: {true_positive}  TN: {true_negative}\n"
            f"FP: {false_positive}  FN: {false_negative}"
        )
        textbox_props = dict(
            boxstyle="round,pad=0.5", edgecolor="black", facecolor="white", alpha=0.8
        )
        ax.text(
            0.02,
            0.95,
            textbox_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=textbox_props,
        )

    axs[0].set_ylabel("Prediction / Actual")
    plt.suptitle("Model Comparison: Linear vs Logistic vs SVM", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/model_comparison.png")

    # === Confusion Matrices ===
    linear_cm = confusion_matrix(y_test, linear_predicted_labels)
    logistic_cm = confusion_matrix(y_test, logistic_predicted_labels)
    svm_cm = confusion_matrix(y_test, svm_predicted_labels)

    # === Plot Confusion Matrices with TP, FP, TN, FN Labels ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    def add_labels(ax, cm):
        """Add TP, FP, TN, FN labels under the numbers."""
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                label = (
                    "TN"
                    if i == 0 and j == 0
                    else (
                        "FP"
                        if i == 0 and j == 1
                        else "FN" if i == 1 and j == 0 else "TP"
                    )
                )
                ax.text(
                    j,
                    i + 0.2,
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

    sns.heatmap(linear_cm, annot=True, fmt="d", cmap="Blues", ax=axs[0], cbar=False)
    axs[0].set_title("Linear Regression Confusion Matrix")
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("Actual")
    axs[0].set_xticklabels(["Negative (Predicted)", "Positive (Predicted)"])
    axs[0].set_yticklabels(["Negative (Actual)", "Positive (Actual)"])
    add_labels(axs[0], linear_cm)

    sns.heatmap(logistic_cm, annot=True, fmt="d", cmap="Blues", ax=axs[1], cbar=False)
    axs[1].set_title("Logistic Regression Confusion Matrix")
    axs[1].set_xlabel("Predicted")
    axs[1].set_ylabel("Actual")
    axs[1].set_xticklabels(["Negative (Predicted)", "Positive (Predicted)"])
    axs[1].set_yticklabels(["Negative (Actual)", "Positive (Actual)"])
    add_labels(axs[1], logistic_cm)

    sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Blues", ax=axs[2], cbar=False)
    axs[2].set_title("SVM Confusion Matrix")
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("Actual")
    axs[2].set_xticklabels(["Negative (Predicted)", "Positive (Predicted)"])
    axs[2].set_yticklabels(["Negative (Actual)", "Positive (Actual)"])
    add_labels(axs[2], svm_cm)

    plt.tight_layout()
    plt.savefig("results/confusion_matrices_with_labels.png")

    # === Print Confusion Matrices ===
    print("\nConfusion Matrices:")
    print("Linear Regression Confusion Matrix:")
    print(linear_cm)
    print("\nLogistic Regression Confusion Matrix:")
    print(logistic_cm)
    print("\nSVM Confusion Matrix:")
    print(svm_cm)

    # === Print final metrics summary ===
    print("\nFinal Model Accuracies:")
    linear_accuracy = np.mean(linear_predicted_labels == y_test)
    logistic_accuracy = np.mean(logistic_predicted_labels == y_test)
    svm_accuracy = np.mean(svm_predicted_labels == y_test)

    print(f"Linear Regression Accuracy: {linear_accuracy * 100:.2f}%")
    print(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")
    print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

    # === Feature Importance Visualization ===
    feature_names = [
        "Gender",
        "Study Hours Per Week",
        "Attendance Rate",
        "Previous Exam Scores",
        "Parental Education Level",
        "Internet Access at Home",
        "Extracurricular Activities",
        "Final Exam Score",
    ]

    # Linear Regression Feature Importance
    linear_coefficients = linear_model.weights
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, linear_coefficients, color="blue", alpha=0.7)
    plt.title("Feature Importance - Linear Regression")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/linear_feature_importance.png")

    # Logistic Regression Feature Importance
    logistic_coefficients = logistic_model.weights
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, logistic_coefficients, color="green", alpha=0.7)
    plt.title("Feature Importance - Logistic Regression")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/logistic_feature_importance.png")

    svm_coefficients = svm_model.weights
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, svm_coefficients, color="red", alpha=0.7)
    plt.title("Feature Importance - SVM")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/svm_feature_importance.png")
