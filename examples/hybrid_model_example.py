"""
Hybrid Classical-Quantum Model Example

Demonstrates building and training a hybrid model that
combines classical preprocessing with quantum circuits.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from quantumcortex import HybridQuantumClassicalModel
from quantumcortex.training import HybridTrainer, Adam


def main():
    print("=" * 60)
    print("Hybrid Classical-Quantum Model Example")
    print("=" * 60)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_train = 100
    n_test = 20
    
    # Create data with nonlinear relationship
    X_train = np.random.randn(n_train, 8)
    y_train = (np.sum(X_train[:, :4], axis=1) + 
               0.5 * np.sin(X_train[:, 4]) +
               0.3 * X_train[:, 5] ** 2 +
               np.random.randn(n_train) * 0.1).reshape(-1, 1)
    
    X_test = np.random.randn(n_test, 8)
    y_test = (np.sum(X_test[:, :4], axis=1) + 
              0.5 * np.sin(X_test[:, 4]) +
              0.3 * X_test[:, 5] ** 2).reshape(-1, 1)
    
    print(f"\nTraining samples: {n_train}")
    print(f"Test samples: {n_test}")
    
    # Create hybrid model
    print("\n" + "-" * 60)
    print("Creating Hybrid Model...")
    
    model = HybridQuantumClassicalModel(
        input_dim=8,
        output_dim=1,
        num_qubits=6,
        num_quantum_layers=2,
        classical_pre_layers=[6],
        classical_post_layers=[4],
        name="HybridRegressor"
    )
    
    print(model.summary())
    
    # Create trainer
    print("\n" + "-" * 60)
    print("Setting up training...")
    
    trainer = HybridTrainer(
        model=model,
        optimizer=Adam(lr=0.01),
        config={
            'epochs': 50,
            'batch_size': 16,
            'verbose': 1
        }
    )
    
    # Train
    print("\n" + "-" * 60)
    print("Training model...")
    
    history = trainer.fit(X_train, y_train, X_test, y_test)
    
    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation:")
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)
    
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, n_test)):
        print(f"  Input: {X_test[i][:4]}...")
        print(f"  Predicted: {test_pred[i, 0]:.3f}")
        print(f"  Actual: {y_test[i, 0]:.3f}")
        print()
    
    print("=" * 60)


if __name__ == "__main__":
    main()
