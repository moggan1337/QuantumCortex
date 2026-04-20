"""
Quantum Kernel Methods Module

Implements quantum kernel functions for quantum machine learning,
including amplitude kernels, basis kernels, and Hilbert-Schmidt kernels.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from quantumcortex.core.quantum_state import QuantumState
from quantumcortex.circuits.vqc import VariationalQuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit


@dataclass
class KernelConfig:
    """Configuration for quantum kernel."""
    num_qubits: int = 4
    feature_map_type: str = 'amplitude'  # 'amplitude', 'basis', 'angle'
    kernel_type: str = 'hilbert_schmidt'  # 'hilbert_schmidt', 'projective'
    normalization: bool = True


class QuantumKernel(ABC):
    """
    Abstract base class for quantum kernels.
    
    A quantum kernel k(x, y) = |⟨φ(x)|φ(y)⟩|² computes
    similarity between data points in a quantum feature space.
    """
    
    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
    
    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute kernel value between two data points.
        
        Args:
            x1: First data point
            x2: Second data point
            
        Returns:
            Kernel value k(x1, x2)
        """
        pass
    
    @abstractmethod
    def encode(self, x: np.ndarray) -> QuantumState:
        """
        Encode data point into quantum state.
        
        Args:
            x: Data point
            
        Returns:
            Quantum state |φ(x)⟩
        """
        pass
    
    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix K where K[i,j] = k(X[i], X[j]).
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Gram matrix (n_samples, n_samples)
        """
        n = len(X)
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                k = self(X[i], X[j])
                K[i, j] = k
                K[j, i] = k
        
        return K
    
    def __repr__(self) -> str:
        return f"QuantumKernel({self.config.kernel_type})"


class AmplitudeKernel(QuantumKernel):
    """
    Amplitude-based Quantum Kernel.
    
    Encodes data into quantum state amplitudes:
    |φ(x)⟩ = Σᵢ fᵢ(x) |i⟩
    
    Kernel: k(x, y) = |Σᵢ fᵢ(x) fᵢ*(y)|² = |⟨ψ(x)|ψ(y)⟩|²
    """
    
    def __init__(self, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.config.feature_map_type = 'amplitude'
    
    def encode(self, x: np.ndarray) -> QuantumState:
        """
        Encode using amplitude embedding.
        
        |φ(x)⟩ = Σᵢ xᵢ/||x|| |i⟩
        """
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        
        # Pad or truncate to match qubit count
        size = 2 ** self.config.num_qubits
        if len(x_norm) < size:
            amplitudes = np.zeros(size, dtype=complex)
            amplitudes[:len(x_norm)] = x_norm
        else:
            amplitudes = x_norm[:size].astype(complex)
        
        # Normalize
        amplitudes = amplitudes / (np.linalg.norm(amplitudes) + 1e-8)
        
        return QuantumState(amplitudes, self.config.num_qubits)
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute amplitude kernel."""
        state1 = self.encode(x1)
        state2 = self.encode(x2)
        
        # Kernel is squared overlap
        overlap = np.abs(np.vdot(state1.state_vector, state2.state_vector)) ** 2
        
        if self.config.normalization:
            norm1 = np.linalg.norm(x1)
            norm2 = np.linalg.norm(x2)
            if norm1 > 0 and norm2 > 0:
                overlap = overlap / (norm1 * norm2)
        
        return np.real(overlap)


class AngleKernel(QuantumKernel):
    """
    Angle-based Quantum Kernel (Basis Encoding).
    
    Encodes data as rotation angles:
    |φ(x)⟩ = ⊗ᵢ R_y(xᵢ) |0⟩
    
    Kernel: k(x, y) = |⟨φ(x)|φ(y)⟩|²
    """
    
    def __init__(self, config: Optional[KernelConfig] = None):
        super().__init__(config)
        self.config.feature_map_type = 'angle'
    
    def encode(self, x: np.ndarray) -> QuantumState:
        """
        Encode using angle encoding.
        
        Each feature xᵢ encodes as rotation R_y(θᵢ) on qubit i.
        """
        from quantumcortex.core.quantum_state import QuantumCircuit
        
        circuit = QuantumCircuit(self.config.num_qubits)
        
        # Normalize features to [0, π]
        x_norm = np.clip(x, -1, 1) * np.pi / 2
        
        # Apply rotations
        for i, theta in enumerate(x_norm[:self.config.num_qubits]):
            circuit.ry(i, theta)
        
        return circuit.execute()
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute angle kernel."""
        state1 = self.encode(x1)
        state2 = self.encode(x2)
        
        overlap = np.abs(np.vdot(state1.state_vector, state2.state_vector)) ** 2
        
        return np.real(overlap)


class HilbertSchmidtKernel(QuantumKernel):
    """
    Hilbert-Schmidt Quantum Kernel.
    
    k(x, y) = ⟨ψ(x)|ψ(y)⟩⟨ψ(y)|ψ(x)⟩ = |⟨ψ(x)|ψ(y)⟩|²
    """
    
    def __init__(
        self,
        feature_map: Optional[Callable] = None,
        config: Optional[KernelConfig] = None
    ):
        super().__init__(config)
        self.config.kernel_type = 'hilbert_schmidt'
        self.feature_map = feature_map or self._default_feature_map
    
    def _default_feature_map(self, x: np.ndarray) -> QuantumState:
        """Default feature map using amplitude encoding."""
        kernel = AmplitudeKernel(self.config)
        return kernel.encode(x)
    
    def encode(self, x: np.ndarray) -> QuantumState:
        """Encode using feature map."""
        return self.feature_map(x)
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Hilbert-Schmidt kernel."""
        state1 = self.encode(x1)
        state2 = self.encode(x2)
        
        # k(x, y) = ⟨ψ(x)|ψ(y)⟩⟨ψ(y)|ψ(x)⟩ = |⟨ψ(x)|ψ(y)⟩|²
        overlap = np.vdot(state1.state_vector, state2.state_vector)
        kernel = np.abs(overlap) ** 2
        
        return np.real(kernel)


class ProjectiveKernel(QuantumKernel):
    """
    Projective Quantum Kernel.
    
    k(x, y) = |⟨0|U(x)†U(y)|0⟩|²
    
    Uses unitary encoding with projective measurement.
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        config: Optional[KernelConfig] = None
    ):
        super().__init__(config)
        self.config.kernel_type = 'projective'
        self.num_layers = num_layers
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, Callable]:
        """
        Encode into unitary parameters.
        
        Returns:
            Tuple of (parameters, unitary_function)
        """
        # Create parameters from data
        params = np.clip(x, -1, 1) * np.pi
        
        def unitary(params):
            """Build unitary from parameters."""
            from quantumcortex.core.quantum_state import QuantumCircuit
            circuit = QuantumCircuit(self.config.num_qubits)
            
            for layer in range(self.num_layers):
                # Parameterized rotations
                for q in range(self.config.num_qubits):
                    idx = (layer * self.config.num_qubits + q) % len(params)
                    circuit.ry(q, params[idx])
                
                # Entangling gates
                for q in range(self.config.num_qubits - 1):
                    circuit.cnot(q, q + 1)
            
            return circuit.get_unitary()
        
        return params, unitary
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute projective kernel."""
        params1, U1 = self.encode(x1)
        params2, U2 = self.encode(x2)
        
        # Projective kernel
        U1_mat = U1(params1)
        U2_mat = U2(params2)
        
        # Compute overlap
        # k = |⟨0|U₁†U₂|0⟩|²
        overlap_mat = np.conj(U1_mat.T) @ U2_mat
        initial = np.zeros(2 ** self.config.num_qubits, dtype=complex)
        initial[0] = 1.0
        
        overlap = np.vdot(initial, overlap_mat @ initial)
        kernel = np.abs(overlap) ** 2
        
        return np.real(kernel)


class VariationalKernel(QuantumKernel):
    """
    Variational Quantum Kernel.
    
    Uses a parameterized quantum circuit as the kernel,
    trained to maximize class separation.
    """
    
    def __init__(
        self,
        num_qubits: int = 4,
        num_layers: int = 2,
        config: Optional[KernelConfig] = None
    ):
        super().__init__(config)
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # Create variational circuit
        self.vqc = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers
        )
    
    def encode(self, x: np.ndarray) -> QuantumState:
        """Encode using variational circuit."""
        # Use VQC's built-in encoding
        state = self.vqc.forward(x)
        return state
    
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute variational kernel."""
        state1 = self.encode(x1)
        state2 = self.encode(x2)
        
        # Use state fidelity as kernel
        fidelity = state1.fidelity(state2)
        
        return fidelity
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01
    ):
        """
        Train kernel to maximize class separation.
        
        Args:
            X: Training data
            y: Labels
            epochs: Training epochs
            lr: Learning rate
        """
        from quantumcortex.training.optimizer import Adam
        
        optimizer = Adam(lr=lr)
        params = self.vqc.parameters
        
        for epoch in range(epochs):
            # Compute kernel matrix
            K = self.gram_matrix(X)
            
            # Kernel target: K_ij = 1 if y_i == y_j, else 0
            target = np.outer(y, y) == 0
            
            # Loss: maximize same-class similarity, minimize different-class
            # Use soft margin loss
            K_safe = np.clip(K, 1e-10, 1)
            loss = np.mean(target * (1 - K_safe) + (1 - target) * K_safe)
            
            # Compute gradients (simplified)
            grad = self._compute_kernel_gradients(X, y)
            
            # Update parameters
            params = optimizer.step(params, grad)
            self.vqc.set_parameters(params)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    def _compute_kernel_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """Compute gradients of kernel w.r.t. circuit parameters."""
        gradients = {}
        eps = 1e-5
        
        for name, value in self.vqc.parameters.items():
            # Numerical gradient
            self.vqc.parameters[name] = value + eps
            K_plus = self.gram_matrix(X)
            
            self.vqc.parameters[name] = value - eps
            K_minus = self.gram_matrix(X)
            
            gradients[name] = np.mean(K_plus - K_minus) / (2 * eps)
            
            self.vqc.parameters[name] = value
        
        return gradients


class QuantumKernelClassifier:
    """
    Quantum Kernel-based Classifier.
    
    Uses a quantum kernel with a classical classifier
    (SVM, logistic regression, etc.) for classification.
    """
    
    def __init__(
        self,
        kernel: Optional[QuantumKernel] = None,
        classifier: str = 'svm',
        C: float = 1.0
    ):
        self.kernel = kernel or HilbertSchmidtKernel()
        self.classifier_type = classifier
        self.C = C
        
        # Kernel matrix
        self.K_train = None
        self.X_train = None
        self.y_train = None
        
        # Classifier parameters
        self.alphas = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the kernel classifier.
        
        Args:
            X: Training data
            y: Training labels
        """
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix
        self.K_train = self.kernel.gram_matrix(X)
        
        if self.classifier_type == 'svm':
            self._fit_svm()
        elif self.classifier_type == 'ridge':
            self._fit_ridge()
    
    def _fit_svm(self):
        """Fit SVM in kernel space."""
        n = len(self.y_train)
        
        # Simplified SVM: use kernel matrix for dual formulation
        # max_α Σ α_i - 0.5 Σ α_i α_j y_i y_j K_ij
        # subject to 0 ≤ α_i ≤ C, Σ α_i y_i = 0
        
        # For simplicity, use direct kernel-based classification
        # Compute class centroids in kernel space
        classes = np.unique(self.y_train)
        
        self.centroids = {}
        for c in classes:
            mask = self.y_train == c
            self.centroids[c] = np.mean(self.K_train[mask], axis=0)
    
    def _fit_ridge(self):
        """Fit kernel ridge regression."""
        n = len(self.y_train)
        
        # Regularized kernel matrix
        K_reg = self.K_train + self.C * np.eye(n)
        
        # Solve (K + λI) α = y
        self.alphas = np.linalg.solve(K_reg, self.y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Test data
            
        Returns:
            Predicted labels
        """
        # Compute kernel with training data
        n_test = len(X)
        n_train = len(self.X_train)
        
        K_test = np.zeros((n_test, n_train))
        for i in range(n_test):
            for j in range(n_train):
                K_test[i, j] = self.kernel(X[i], self.X_train[j])
        
        if self.classifier_type == 'svm':
            # Nearest centroid
            predictions = []
            for i in range(n_test):
                distances = {
                    c: np.linalg.norm(K_test[i] - centroid)
                    for c, centroid in self.centroids.items()
                }
                predictions.append(min(distances, key=distances.get))
            return np.array(predictions)
        
        elif self.classifier_type == 'ridge':
            return K_test @ self.alphas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def create_kernel(
    kernel_type: str,
    num_qubits: int = 4,
    **kwargs
) -> QuantumKernel:
    """
    Factory function to create quantum kernels.
    
    Args:
        kernel_type: Type of kernel
        num_qubits: Number of qubits
        **kwargs: Additional arguments
        
    Returns:
        QuantumKernel instance
    """
    config = KernelConfig(num_qubits=num_qubits, **kwargs)
    
    kernels = {
        'amplitude': AmplitudeKernel,
        'angle': AngleKernel,
        'hilbert_schmidt': HilbertSchmidtKernel,
        'projective': ProjectiveKernel,
        'variational': VariationalKernel,
    }
    
    if kernel_type.lower() not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernels[kernel_type.lower()](config)
