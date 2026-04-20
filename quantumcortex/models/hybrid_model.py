"""
Hybrid Quantum-Classical Neural Network Model

Implements models that combine classical neural network layers
with quantum circuits for end-to-end learning.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

from quantumcortex.circuits.vqc import VariationalQuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit
from quantumcortex.layers.quantum_perceptron import (
    QuantumDenseLayer, QuantumEmbeddingLayer, QuantumAttention
)
from quantumcortex.layers.quantum_conv import (
    QuantumConvolutionalLayer, QuantumPoolingLayer, Quantum1DConvolution
)
from quantumcortex.layers.quantum_recurrent import (
    QuantumRecurrentLayer, QuantumLSTMLayer, QuantumGRUCell
)


@dataclass
class LayerConfig:
    """Configuration for a single layer in the hybrid model."""
    layer_type: str
    params: Dict[str, Any]


class HybridModel(ABC):
    """
    Abstract base class for hybrid quantum-classical models.
    
    Defines the interface for models that combine classical
    and quantum components.
    """
    
    def __init__(self, name: str = "HybridModel"):
        self.name = name
        self.layers: List[Any] = []
        self._is_compiled = False
        self._loss_history: List[float] = []
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients."""
        pass
    
    def compile(
        self,
        loss: Optional[Callable] = None,
        optimizer: Optional[Any] = None
    ):
        """Compile the model for training."""
        self.loss = loss or self._default_loss
        self.optimizer = optimizer
        self._is_compiled = True
    
    def _default_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default MSE loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X: Training inputs
            y: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional (X_val, y_val)
            verbose: Print progress
            
        Returns:
            Training history
        """
        if not self._is_compiled:
            self.compile()
        
        history = {'loss': [], 'val_loss': [] if validation_data else None}
        
        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss(y_batch, y_pred)
                epoch_loss += loss
                
                # Backward pass
                self.backward(y_batch - y_pred)
            
            epoch_loss /= n_batches
            self._loss_history.append(epoch_loss)
            history['loss'].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                val_pred = self.forward(validation_data[0])
                val_loss = self.loss(validation_data[1], val_pred)
                history['val_loss'].append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class HybridQuantumClassicalModel(HybridModel):
    """
    Flexible Hybrid Quantum-Classical Model.
    
    Combines classical layers (dense, conv, recurrent) with
    quantum circuits in a configurable architecture.
    
    Example architecture:
        Input -> Classical Preprocessing -> Quantum Layer -> Classical Postprocessing -> Output
    
    Attributes:
        layer_configs: List of layer configurations
        quantum_layer: The quantum circuit component
        classical_pre: Classical preprocessing layers
        classical_post: Classical postprocessing layers
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_qubits: int = 4,
        num_quantum_layers: int = 2,
        classical_pre_layers: Optional[List[int]] = None,
        classical_post_layers: Optional[List[int]] = None,
        name: str = "HybridQCNN"
    ):
        super().__init__(name)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_qubits = num_qubits
        
        # Classical preprocessing layers
        self.classical_pre: List[QuantumDenseLayer] = []
        if classical_pre_layers:
            prev_dim = input_dim
            for hidden_dim in classical_pre_layers:
                self.classical_pre.append(
                    QuantumDenseLayer(prev_dim, hidden_dim, num_qubits=num_qubits)
                )
                prev_dim = hidden_dim
            self.pre_output_dim = prev_dim
        else:
            self.pre_output_dim = input_dim
        
        # Quantum layer
        self.quantum_layer = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_quantum_layers,
            ansatz_type='hardware_efficient'
        )
        
        # Classical postprocessing layers
        self.classical_post: List[QuantumDenseLayer] = []
        if classical_post_layers:
            # Input to quantum layer is reduced dimension
            quantum_input_dim = num_qubits
            prev_dim = quantum_input_dim
            for hidden_dim in classical_post_layers:
                self.classical_post.append(
                    QuantumDenseLayer(prev_dim, hidden_dim, num_qubits=num_qubits)
                )
                prev_dim = hidden_dim
            # Final layer to output_dim
            self.classical_post.append(
                QuantumDenseLayer(prev_dim, output_dim, num_qubits=num_qubits)
            )
        else:
            # Simple quantum to output projection
            self.classical_post = []
        
        # Store parameters for optimization
        self._collect_parameters()
    
    def _collect_parameters(self):
        """Collect all trainable parameters."""
        self.parameters = {}
        
        # Classical pre parameters
        for i, layer in enumerate(self.classical_pre):
            if hasattr(layer, 'weights'):
                self.parameters[f'pre_{i}_weights'] = layer.weights
            if hasattr(layer, 'biases'):
                self.parameters[f'pre_{i}_biases'] = layer.biases
        
        # Quantum parameters
        self.parameters['quantum'] = self.quantum_layer.parameters
        
        # Classical post parameters
        for i, layer in enumerate(self.classical_post):
            if hasattr(layer, 'weights'):
                self.parameters[f'post_{i}_weights'] = layer.weights
            if hasattr(layer, 'biases'):
                self.parameters[f'post_{i}_biases'] = layer.biases
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input (batch_size, input_dim)
            
        Returns:
            Output (batch_size, output_dim)
        """
        # Classical preprocessing
        h = x
        for layer in self.classical_pre:
            h = layer.forward(h)
        
        # Encode to quantum layer dimension
        quantum_input_size = min(len(h[0]), self.num_qubits)
        h_reduced = h[:, :quantum_input_size]
        
        # Normalize for quantum encoding
        h_norm = h_reduced / (np.linalg.norm(h_reduced, axis=1, keepdims=True) + 1e-8)
        
        # Quantum processing
        quantum_outputs = []
        for sample in h_norm:
            state = self.quantum_layer.forward(sample)
            probs = np.abs(state.state_vector) ** 2
            quantum_outputs.append(probs[:self.num_qubits])
        
        h_quantum = np.array(quantum_outputs)
        
        # Classical postprocessing
        output = h_quantum
        for i, layer in enumerate(self.classical_post):
            output = layer.forward(output)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through hybrid model.
        
        For hybrid models, gradients are computed numerically
        or using the parameter shift rule on quantum parameters.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Gradient w.r.t. input
        """
        # Numerical gradient estimation for simplicity
        # In practice, would use analytic gradients via parameter shift
        
        grad_input = np.zeros_like(grad_output)
        
        # This is a simplified backward pass
        # Full implementation would track all parameter gradients
        
        return grad_input
    
    def compute_quantum_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shots: int = 1000
    ) -> Dict[str, float]:
        """
        Compute gradients for quantum parameters using parameter shift.
        
        Args:
            X: Input data
            y: Target data
            shots: Number of measurement shots
            
        Returns:
            Dictionary of parameter gradients
        """
        gradients = {}
        h = np.pi / 2
        
        for param_name, param_value in self.quantum_layer.parameters.items():
            # Shifted forward passes
            params_plus = self.quantum_layer.parameters.copy()
            params_plus[param_name] = param_value + h
            
            y_pred_plus = self.forward_with_params(X, params_plus)
            loss_plus = np.mean((y - y_pred_plus) ** 2)
            
            params_minus = self.quantum_layer.parameters.copy()
            params_minus[param_name] = param_value - h
            
            y_pred_minus = self.forward_with_params(X, params_minus)
            loss_minus = np.mean((y - y_pred_minus) ** 2)
            
            # Parameter shift gradient
            gradients[param_name] = (loss_plus - loss_minus) / 2
        
        return gradients
    
    def forward_with_params(
        self,
        x: np.ndarray,
        quantum_params: Dict[str, float]
    ) -> np.ndarray:
        """Forward pass with specific quantum parameters."""
        # Temporarily set parameters
        original_params = self.quantum_layer.parameters.copy()
        self.quantum_layer.set_parameters(quantum_params)
        
        output = self.forward(x)
        
        # Restore original parameters
        self.quantum_layer.set_parameters(original_params)
        
        return output
    
    def summary(self) -> str:
        """Print model summary."""
        lines = [f"Hybrid Quantum-Classical Model: {self.name}"]
        lines.append(f"  Input dimension: {self.input_dim}")
        lines.append(f"  Output dimension: {self.output_dim}")
        lines.append(f"  Number of qubits: {self.num_qubits}")
        lines.append("\n  Classical Preprocessing:")
        for i, layer in enumerate(self.classical_pre):
            lines.append(f"    Layer {i}: {layer}")
        lines.append(f"\n  Quantum Layer:")
        lines.append(f"    {self.quantum_layer}")
        lines.append("\n  Classical Postprocessing:")
        for i, layer in enumerate(self.classical_post):
            lines.append(f"    Layer {i}: {layer}")
        return "\n".join(lines)


class HybridCNNQNN(HybridModel):
    """
    Hybrid CNN-QNN model for image classification.
    
    Architecture:
        Input (H, W, C) -> Classical Conv/Pool -> Flatten -> Quantum Dense -> Output
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        num_qubits: int = 6,
        conv_filters: List[int] = None,
        name: str = "HybridCNNQNN"
    ):
        super().__init__(name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Default convolutional architecture
        if conv_filters is None:
            conv_filters = [16, 32]
        
        # Convolutional layers
        self.conv_layers: List[QuantumConvolutionalLayer] = []
        in_channels = input_shape[2]
        
        for filters in conv_filters:
            self.conv_layers.append(
                QuantumConvolutionalLayer(
                    input_channels=in_channels,
                    output_channels=filters,
                    config=None
                )
            )
            in_channels = filters
        
        # Pooling
        self.pool = QuantumPoolingLayer(pool_size=(2, 2))
        
        # Calculate flattened size
        # Simplified: assume output after conv/pool is small
        self.flatten_size = conv_filters[-1] * 4 * 4 if conv_filters else 64
        
        # Quantum dense layer
        self.quantum_dense = QuantumDenseLayer(
            self.flatten_size,
            num_classes,
            num_qubits=num_qubits
        )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Conv layers
        h = x
        for conv in self.conv_layers:
            h = conv.forward(h)
            h = self.pool.forward(h)
        
        # Flatten
        batch_size = h.shape[0]
        h = h.reshape(batch_size, -1)
        
        # Take only what fits in quantum layer
        h = h[:, :self.flatten_size]
        
        # Quantum dense
        output = self.quantum_dense.forward(h)
        
        # Softmax
        output = np.exp(output - np.max(output, axis=1, keepdims=True))
        output = output / output.sum(axis=1, keepdims=True)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # Simplified
        return grad_output


class HybridRNNQNN(HybridModel):
    """
    Hybrid RNN-QNN model for sequence modeling.
    
    Architecture:
        Input (seq_len, features) -> Classical RNN/LSTM -> Quantum Dense -> Output
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_qubits: int = 6,
        rnn_type: str = 'lstm',
        num_layers: int = 1,
        name: str = "HybridRNNQNN"
    ):
        super().__init__(name)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # RNN layers
        self.rnn_layers: List[Any] = []
        for i in range(num_layers):
            is_first = i == 0
            rnn_input_size = input_size if is_first else hidden_size
            
            if rnn_type == 'lstm':
                self.rnn_layers.append(
                    QuantumLSTMLayer(rnn_input_size, hidden_size, num_qubits=num_qubits)
                )
            else:
                self.rnn_layers.append(
                    QuantumRecurrentLayer(
                        rnn_input_size, hidden_size,
                        config=QuantumRNNConfig(num_qubits=num_qubits)
                    )
                )
        
        # Quantum output layer
        self.quantum_output = QuantumDenseLayer(
            hidden_size,
            num_classes,
            num_qubits=num_qubits
        )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # RNN processing
        h = x
        hidden_states = None
        cell_states = None
        
        for layer in self.rnn_layers:
            if isinstance(layer, QuantumLSTMLayer):
                h, (hidden_states, cell_states) = layer.forward(h)
            else:
                h, hidden_states = layer.forward(h)
        
        # Take last output
        if isinstance(h, tuple):
            h = h[0]
        last_output = h[:, -1, :] if h.ndim == 3 else h
        
        # Quantum output
        output = self.quantum_output.forward(last_output)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad_output
