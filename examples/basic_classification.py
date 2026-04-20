"""
Basic Classification Example with QuantumCortex

This example demonstrates how to use QuantumCortex for
binary classification using a quantum neural network.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from quantumcortex import QNNClassifier


def generate_synthetic_data(n_samples=200, seed=42):
    """Generate synthetic classification data."""
    np.random.seed(seed)
    
    # Class 0: centered at (0, 0)
    X0 = np.random.randn(n_samples // 2, 4) * 0.5 + np.array([0, 0, 0, 0])
    
    # Class 1: centered at (1, 1, 1, 1)
    X1 = np.random.randn(n_samples // 2, 4) * 0.5 + np.array([1, 1, 1, 1])
    
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def main():
    print("=" * 60)
    print("QuantumCortex Basic Classification Example")
    print("=" * 60)
    
    # Generate data
    X_train, y_train = generate_synthetic_data(n_samples=200)
    X_test, y_test = generate_synthetic_data(n_samples=50, seed=123)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Create classifier
    print("\n" + "-" * 60)
    print("Creating QNN Classifier...")
    
    clf = QNNClassifier(
        input_dim=4,
        num_classes=2,
        num_qubits=4,
        num_layers=2,
        encoding_method='angle'
    )
    
    print(f"Classifier created with {clf.num_qubits} qubits")
    
    # Train
    print("\n" + "-" * 60)
    print("Training...")
    
    history = clf.fit(X_train, y_train, epochs=50, verbose=True)
    
    # Evaluate
    print("\n" + "-" * 60)
    print("Evaluating...")
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions
    print("\n" + "-" * 60)
    print("Sample Predictions:")
    
    for i in range(min(5, len(X_test))):
        prob = clf.predict_proba(X_test[i:i+1])[0]
        print(f"  Sample {i+1}: Pred={test_pred[i]}, "
              f"Probs=[{prob[0]:.3f}, {prob[1]:.3f}], "
              f"True={y_test[i]}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
