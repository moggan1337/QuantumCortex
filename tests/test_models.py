"""
Tests for quantum models.
"""

import numpy as np
import pytest
from quantumcortex.models import (
    HybridQuantumClassicalModel,
    QNNClassifier,
    QNNRegressor
)


class TestHybridModel:
    """Test cases for hybrid quantum-classical models."""
    
    def test_create_hybrid_model(self):
        """Test hybrid model creation."""
        model = HybridQuantumClassicalModel(
            input_dim=10,
            output_dim=1,
            num_qubits=4,
            num_quantum_layers=2
        )
        
        assert model.input_dim == 10
        assert model.output_dim == 1
        assert model.num_qubits == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = HybridQuantumClassicalModel(
            input_dim=8,
            output_dim=2,
            num_qubits=4,
            num_quantum_layers=1
        )
        
        x = np.random.randn(4, 8)
        output = model.forward(x)
        
        assert output.shape == (4, 2)
    
    def test_predict(self):
        """Test prediction."""
        model = HybridQuantumClassicalModel(
            input_dim=4,
            output_dim=1,
            num_qubits=4
        )
        
        x = np.random.randn(10, 4)
        predictions = model.predict(x)
        
        assert predictions.shape == (10, 1)


class TestQNNClassifier:
    """Test cases for QNN classifier."""
    
    def test_create_classifier(self):
        """Test classifier creation."""
        clf = QNNClassifier(
            input_dim=4,
            num_classes=2,
            num_qubits=4
        )
        
        assert clf.num_classes == 2
    
    def test_forward(self):
        """Test forward pass."""
        clf = QNNClassifier(
            input_dim=4,
            num_classes=3,
            num_qubits=4
        )
        
        x = np.random.randn(5, 4)
        probs = clf.forward(x)
        
        assert probs.shape == (5, 3)
        # Check probabilities sum to 1
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=0.1)
    
    def test_predict(self):
        """Test prediction."""
        clf = QNNClassifier(
            input_dim=4,
            num_classes=2,
            num_qubits=4
        )
        
        x = np.random.randn(10, 4)
        predictions = clf.predict(x)
        
        assert predictions.shape == (10,)
        assert set(predictions).issubset({0, 1})


class TestQNNRegressor:
    """Test cases for QNN regressor."""
    
    def test_create_regressor(self):
        """Test regressor creation."""
        reg = QNNRegressor(
            input_dim=4,
            num_outputs=1,
            num_qubits=4
        )
        
        assert reg.num_outputs == 1
    
    def test_forward(self):
        """Test forward pass."""
        reg = QNNRegressor(
            input_dim=4,
            num_outputs=1,
            num_qubits=4
        )
        
        x = np.random.randn(5, 4)
        output = reg.forward(x)
        
        assert output.shape == (5, 1)
    
    def test_loss_computation(self):
        """Test loss computation."""
        reg = QNNRegressor(input_dim=4, num_outputs=1)
        
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.1], [1.9], [3.2]])
        
        loss = reg.compute_loss(y_true, y_pred)
        
        assert loss > 0
        assert loss < 0.1  # Small error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
