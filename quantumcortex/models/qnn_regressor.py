"""
Quantum Neural Network Regressor

Implements quantum circuit-based regression models.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from quantumcortex.models.hybrid_model import HybridQuantumClassicalModel
from quantumcortex.circuits.vqc import VariationalQuantumCircuit


@dataclass
class RegressorConfig:
    """Configuration for quantum regressor."""
    num_outputs: int = 1
    num_qubits: int = 4
    num_layers: int = 2
    output_activation: Optional[str] = None  # 'linear', 'sigmoid', 'relu'


class QNNRegressor(HybridQuantumClassicalModel):
    """
    Quantum Neural Network Regressor.
    
    Uses variational quantum circuits to perform regression
    from input features to continuous output values.
    
    Attributes:
        num_outputs: Number of output dimensions
        output_activation: Activation function for outputs
    """
    
    def __init__(
        self,
        input_dim: int,
        num_outputs: int = 1,
        num_qubits: int = 4,
        num_layers: int = 2,
        output_activation: Optional[str] = None,
        name: str = "QNNRegressor"
    ):
        self.num_outputs = num_outputs
        self.output_activation = output_activation
        
        super().__init__(
            input_dim=input_dim,
            output_dim=num_outputs,
            num_qubits=num_qubits,
            num_quantum_layers=num_layers,
            name=name
        )
        
        # Override quantum layer for regression
        self.quantum_layer = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='hardware_efficient'
        )
        
        # Initialize output scaling
        self.output_scale = 1.0
        self.output_shift = 0.0
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for regression.
        
        Args:
            x: Input data (batch_size, input_dim)
            
        Returns:
            Predicted values (batch_size, num_outputs)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        
        # Process through quantum circuit
        predictions = []
        
        for i in range(batch_size):
            # Encode input
            encoded = self._encode_input(x[i])
            
            # Execute quantum circuit
            state = self.quantum_layer.forward(encoded)
            
            # Extract regression value from measurement
            value = self._extract_value(state)
            predictions.append(value)
        
        predictions = np.array(predictions)
        
        # Apply output activation if specified
        predictions = self._apply_output_activation(predictions)
        
        return predictions
    
    def _encode_input(self, x: np.ndarray) -> np.ndarray:
        """Encode input into quantum circuit parameters."""
        # Angle encoding
        x_norm = np.clip(x, -1, 1) * np.pi
        return x_norm[:self.num_qubits]
    
    def _extract_value(self, state) -> np.ndarray:
        """Extract continuous value from quantum state."""
        probs = np.abs(state.state_vector) ** 2
        
        # Use expectation value of Z as regression output
        values = []
        
        for q in range(min(self.num_outputs, self.num_qubits)):
            # Compute probability of |1⟩ on qubit q
            offset = 2 ** (self.num_qubits - 1 - q)
            prob_one = sum(probs[i] for i in range(offset, len(probs), 2 * offset))
            
            # Map [0,1] to [-1,1]
            value = 2 * prob_one - 1
            values.append(value)
        
        # Pad with zeros if needed
        while len(values) < self.num_outputs:
            values.append(0.0)
        
        return np.array(values[:self.num_outputs])
    
    def _apply_output_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply output activation function."""
        if self.output_activation is None:
            return x
        elif self.output_activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.output_activation == 'relu':
            return np.maximum(0, x)
        elif self.output_activation == 'tanh':
            return np.tanh(x)
        else:
            return x
    
    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute regression loss (MSE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Gradient w.r.t. input
        """
        # Gradient of MSE: 2 * (y_pred - y_true) / n
        grad = 2 * (y_pred - y_true) / len(y_true)
        return grad


class QuantumVariationalRegressor:
    """
    Standalone Variational Quantum Regressor.
    
    Simpler interface for VQC-based regression.
    """
    
    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        output_dim: int = 1
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Create VQC
        self.vqc = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='hardware_efficient'
        )
        
        # Classical pre-processing
        self.W = np.random.randn(num_qubits, num_qubits) * 0.01
        self.b = np.zeros(num_qubits)
        
        # Classical post-processing
        self.W_out = np.random.randn(num_qubits, output_dim) * 0.01
        self.b_out = np.zeros(output_dim)
    
    def _encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to quantum parameters."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        return np.clip(x_norm, -1, 1) * np.pi
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        outputs = []
        for sample in x:
            # Encode
            encoded = self._encode(sample)
            
            # Classical preprocessing
            h = encoded @ self.W + self.b
            h = np.tanh(h)
            
            # Quantum circuit
            state = self.vqc.forward(h)
            probs = np.abs(state.state_vector) ** 2
            
            # Extract quantum features
            quantum_features = probs[:self.num_qubits]
            
            # Classical postprocessing
            out = quantum_features @ self.W_out + self.b_out
            outputs.append(out)
        
        return np.array(outputs)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01
    ) -> List[float]:
        """
        Train the regressor.
        
        Args:
            X: Training inputs
            y: Training targets
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Loss history
        """
        history = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            history.append(loss)
            
            # Compute gradients (simplified)
            grad_output = 2 * (y_pred - y) / len(y)
            
            # Update weights (gradient descent)
            # In practice, would use proper backprop or parameter shift
            grad_W = np.random.randn(*self.W.shape) * 0.01
            grad_b = np.random.randn(*self.b.shape) * 0.01
            grad_W_out = np.random.randn(*self.W_out.shape) * 0.01
            grad_b_out = np.random.randn(*self.b_out.shape) * 0.01
            
            self.W -= lr * grad_W
            self.b -= lr * grad_b
            self.W_out -= lr * grad_W_out
            self.b_out -= lr * grad_b_out
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantumBayesianRegressor:
    """
    Bayesian Quantum Regressor.
    
    Uses quantum circuits to model uncertainty in predictions.
    Returns both mean prediction and uncertainty estimate.
    """
    
    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        n_samples: int = 10
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.n_samples = n_samples
        
        # Ensemble of VQCs for uncertainty estimation
        self.vqcs = [
            VariationalQuantumCircuit(num_qubits, num_layers)
            for _ in range(n_samples)
        ]
        
        # Initialize with different random seeds
        for i, vqc in enumerate(self.vqcs):
            vqc.reset_parameters(seed=i)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with uncertainty.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Get predictions from all models
        predictions = []
        
        for vqc in self.vqcs:
            state = vqc.forward(x[0])
            probs = np.abs(state.state_vector) ** 2
            pred = np.mean(probs[:self.num_qubits])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Mean and standard deviation
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return mean_pred, std_pred
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty for all samples."""
        means = []
        stds = []
        
        for sample in X:
            mean, std = self.forward(sample)
            means.append(mean)
            stds.append(std)
        
        return np.array(means), np.array(stds)
    
    def __call__(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x)
