"""
Quantum Perceptron Layer

Implements quantum versions of the classical perceptron neuron,
including parameterized quantum circuits that mimic perceptron behavior.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from quantumcortex.core.quantum_state import QuantumState, QuantumCircuit
from quantumcortex.circuits.vqc import VariationalQuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit


class ActivationFunction(ABC):
    """Abstract base class for quantum activation functions."""
    
    @abstractmethod
    def __call__(self, x: float) -> float:
        """Apply activation function."""
        pass
    
    @abstractmethod
    def derivative(self, x: float) -> float:
        """Compute derivative."""
        pass


class SigmoidActivation(ActivationFunction):
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""
    
    def __call__(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def derivative(self, x: float) -> float:
        s = self(x)
        return s * (1 - s)


class TanhActivation(ActivationFunction):
    """Hyperbolic tangent activation."""
    
    def __call__(self, x: float) -> float:
        return np.tanh(x)
    
    def derivative(self, x: float) -> float:
        return 1.0 - np.tanh(x) ** 2


class ReLUActivation(ActivationFunction):
    """ReLU activation: max(0, x)"""
    
    def __call__(self, x: float) -> float:
        return max(0, x)
    
    def derivative(self, x: float) -> float:
        return 1.0 if x > 0 else 0.0


class SoftmaxActivation(ActivationFunction):
    """Softmax activation for multi-class output."""
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # Jacobian of softmax
        s = self(x)
        return np.diag(s) - np.outer(s, s)


@dataclass
class QuantumPerceptronConfig:
    """Configuration for a quantum perceptron."""
    num_qubits: int = 4
    num_layers: int = 2
    encoding_qubits: int = 2
    measure_qubits: int = 2
    activation: Optional[str] = 'sigmoid'
    measurement_basis: str = 'z'
    use_entanglement: bool = True


class QuantumPerceptron:
    """
    Quantum Perceptron implementing a single neuron using quantum circuits.
    
    The quantum perceptron uses a parameterized quantum circuit to compute
    a nonlinear transformation of input data, mimicking classical perceptron
    behavior but leveraging quantum effects for potential advantages.
    
    Attributes:
        config: Configuration parameters
        vqc: Variational quantum circuit
        weights: Input weights (classical preprocessing)
        bias: Bias term
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        config: Optional[QuantumPerceptronConfig] = None,
        name: str = "QuantumPerceptron"
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.config = config or QuantumPerceptronConfig()
        
        # Classical pre-processing weights
        self.weights = np.random.randn(input_dim, self.config.encoding_qubits) * 0.1
        self.bias = np.zeros(self.config.encoding_qubits)
        
        # Create VQC for the perceptron
        self.vqc = VariationalQuantumCircuit(
            num_qubits=self.config.num_qubits,
            num_layers=self.config.num_layers,
            ansatz_type='hardware_efficient'
        )
        
        # Set up measurement observable
        self._setup_measurement()
        
        # Activation function
        self.activation = self._get_activation()
    
    def _get_activation(self) -> ActivationFunction:
        """Get activation function."""
        activations = {
            'sigmoid': SigmoidActivation(),
            'tanh': TanhActivation(),
            'relu': ReLUActivation(),
            'softmax': SoftmaxActivation(),
            None: None
        }
        return activations.get(self.config.activation)
    
    def _setup_measurement(self):
        """Set up measurement observable."""
        # Measure expectation value of Z on output qubits
        for i in range(self.config.measure_qubits):
            self.vqc.add_observable('Z' * (self.config.num_qubits - i - 1) + 
                                   'Z' + 'I' * i)
    
    def encode_input(self, x: np.ndarray) -> np.ndarray:
        """
        Encode classical input into quantum parameters.
        
        Args:
            x: Input vector
            
        Returns:
            Encoded parameters for quantum circuit
        """
        # Apply classical preprocessing
        x_processed = x @ self.weights + self.bias
        
        # Normalize to [0, π] range for rotations
        x_encoded = np.clip(x_processed, -1, 1) * np.pi / 2
        
        return x_encoded
    
    def forward(
        self,
        x: np.ndarray,
        parameters: Optional[dict] = None
    ) -> np.ndarray:
        """
        Forward pass through quantum perceptron.
        
        Args:
            x: Input data (batch_size, input_dim)
            parameters: Optional VQC parameters
            
        Returns:
            Output (batch_size, output_dim)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        outputs = np.zeros((batch_size, self.output_dim))
        
        for i in range(batch_size):
            # Encode input
            x_encoded = self.encode_input(x[i])
            
            # Execute VQC
            if parameters is None:
                state = self.vqc.forward(x_encoded)
            else:
                state = self.vqc.forward(x_encoded, parameters)
            
            # Extract output from measurement
            # Use last few qubits as output
            probs = np.abs(state.state_vector) ** 2
            
            # Simple extraction: measure specific qubits
            output_val = 0.0
            for j in range(min(self.output_dim, self.config.measure_qubits)):
                # Get probability of |1⟩ on qubit j
                idx_base = 2 ** (self.config.num_qubits - 1 - j)
                prob_one = sum(probs[idx_base::2 * idx_base])
                output_val += prob_one
            
            output_val /= self.config.measure_qubits
            
            # Apply activation if specified
            if self.activation is not None:
                output_val = self.activation(output_val)
            
            outputs[i, 0] = output_val
        
        return outputs
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Allow calling the perceptron as a function."""
        return self.forward(x)


class QuantumLayer:
    """
    Base class for quantum neural network layers.
    
    Provides common functionality for building quantum layers
    with trainable parameters.
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_parameters: int,
        name: str = "QuantumLayer"
    ):
        self.num_qubits = num_qubits
        self.num_parameters = num_parameters
        self.name = name
        
        # Initialize trainable parameters
        self.parameters = np.random.randn(num_parameters) * 0.01
        
        # Build the circuit template
        self._build_circuit()
    
    @abstractmethod
    def _build_circuit(self):
        """Build the quantum circuit for this layer."""
        pass
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        pass
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantumDenseLayer(QuantumLayer):
    """
    Quantum dense (fully connected) layer.
    
    Uses a variational quantum circuit to implement
    a dense layer transformation.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_qubits: int = 4,
        name: str = "QuantumDense"
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        super().__init__(
            num_qubits=num_qubits,
            num_parameters=num_qubits * output_dim,
            name=name
        )
    
    def _build_circuit(self):
        """Build VQC for dense transformation."""
        self.vqc = VariationalQuantumCircuit(
            num_qubits=self.num_qubits,
            num_layers=1,
            ansatz_type='hardware_efficient'
        )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, input_dim)
            
        Returns:
            Output (batch_size, output_dim)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Encode input
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        
        # Apply quantum transformation
        outputs = []
        for sample in x_norm:
            # Encode into circuit
            state = self.vqc.forward(sample[:self.num_qubits], dict(zip(
                sorted(self.vqc.parameters.keys()),
                self.parameters
            )))
            
            # Extract measurement result
            probs = np.abs(state.state_vector) ** 2
            outputs.append(probs[:self.output_dim])
        
        return np.array(outputs)


class QuantumEmbeddingLayer(QuantumLayer):
    """
    Quantum embedding layer for encoding discrete inputs.
    
    Uses amplitude encoding to embed input vectors into quantum states.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_qubits: Optional[int] = None,
        name: str = "QuantumEmbedding"
    ):
        # Need enough qubits to represent vocab_size amplitudes
        if num_qubits is None:
            num_qubits = int(np.ceil(np.log2(vocab_size)))
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_qubits = num_qubits
        
        super().__init__(
            num_qubits=num_qubits,
            num_parameters=vocab_size * embedding_dim,
            name=name
        )
        
        # Embedding matrix
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def _build_circuit(self):
        """Build embedding preparation circuit."""
        # For amplitude encoding, we need to set up state preparation
        pass
    
    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            indices: Input indices (batch_size,)
            
        Returns:
            Embeddings (batch_size, embedding_dim)
        """
        return self.embedding[indices]
    
    def __call__(self, indices: np.ndarray) -> np.ndarray:
        return self.forward(indices)


class QuantumAttention(QuantumLayer):
    """
    Quantum attention mechanism.
    
    Uses quantum circuits to compute attention weights
    between query and key vectors.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_qubits: int = 6,
        name: str = "QuantumAttention"
    ):
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Attention dimension
        self.attention_dim = min(query_dim, key_dim)
        
        super().__init__(
            num_qubits=num_qubits,
            num_parameters=query_dim * self.attention_dim + 
                         key_dim * self.attention_dim,
            name=name
        )
    
    def _build_circuit(self):
        """Build attention circuit."""
        self.vqc_query = VariationalQuantumCircuit(
            num_qubits=self.num_qubits,
            num_layers=1
        )
        self.vqc_key = VariationalQuantumCircuit(
            num_qubits=self.num_qubits,
            num_layers=1
        )
    
    def forward(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            query: (batch, query_dim)
            keys: (batch, seq_len, key_dim)
            values: (batch, seq_len, value_dim)
            mask: Optional attention mask
            
        Returns:
            Context vector (batch, value_dim)
        """
        # Compute attention scores using quantum circuit
        scores = self._compute_attention_scores(query, keys)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax over scores
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
        
        # Weighted sum of values
        context = np.einsum('bij,bjk->bik', attention_weights, values)
        
        return context.squeeze(1) if context.shape[1] == 1 else context
    
    def _compute_attention_scores(
        self,
        query: np.ndarray,
        keys: np.ndarray
    ) -> np.ndarray:
        """Compute attention scores using quantum circuit."""
        # Simplified: classical dot product
        # Full implementation would use quantum circuit
        scores = np.einsum('bi,bkj->bj', query, keys) / np.sqrt(self.attention_dim)
        return scores
