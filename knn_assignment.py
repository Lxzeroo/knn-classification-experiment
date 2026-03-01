import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def generate_data(seed=42):
    """Generate 100 random values in [0,1] and assign labels."""
    np.random.seed(seed)
    x = np.random.uniform(0, 1, 100)
    y = np.where(x <= 0.5, 1, 2)  # Threshold rule
    return x, y


def split_data(x, y):
    """Split first 50 for training and remaining 50 for testing."""
    x_train = x[:50].reshape(-1, 1)
    y_train = y[:50]
    x_test = x[50:].reshape(-1, 1)
    y_test = y[50:]
    return x_train, y_train, x_test, y_test


def run_knn(x_train, y_train, x_test, y_test, k_values):
    """Train and evaluate KNN for different k values."""
    accuracies = []
    print("\nKNN Classification Results")
    print("-" * 40)
    print(f"{'k':<5}{'Accuracy (%)':>15}")
    print("-" * 40)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred) * 100
        accuracies.append(acc)
        print(f"{k:<5}{acc:>15.2f}")

    print("-" * 40)
    return accuracies


def save_accuracy_plot(k_values, accuracies, output_folder):
    """Save Accuracy vs k graph."""
    plt.figure()
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel("Value of k")
    plt.ylabel("Classification Accuracy (%)")
    plt.title("KNN Accuracy vs k")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "accuracy_vs_k.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def save_training_plot(x_train, y_train, output_folder):
    """Save Training Data Distribution graph."""
    plt.figure()
    plt.scatter(x_train, y_train, c=y_train)
    plt.xlabel("x values")
    plt.ylabel("Class Label")
    plt.title("Training Data Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "training_distribution.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # Create outputs folder automatically
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Generate and split data
    x, y = generate_data()
    x_train, y_train, x_test, y_test = split_data(x, y)

    # Define k values
    k_values = [1, 2, 3, 4, 5, 20, 30]

    # Run KNN
    accuracies = run_knn(x_train, y_train, x_test, y_test, k_values)

    # Save graphs
    save_accuracy_plot(k_values, accuracies, output_folder)
    save_training_plot(x_train, y_train, output_folder)

    print("\nGraphs saved inside 'outputs' folder.")


if __name__ == "__main__":
    main()