"""
Quantum Neural Network Classifier

Implements quantum circuit-based classifiers for
classification tasks.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from quantumcortex.models.hybrid_model import HybridQuantumClassicalModel
from quantumcortex.circuits.vqc import VariationalQuantumCircuit


@dataclass
class ClassifierConfig:
    """Configuration for quantum classifier."""
    num_classes: int
    num_qubits: int = 4
    num_layers: int = 2
    encoding_method: str = 'angle'  # 'angle', 'amplitude', 'basis'
    measurement_strategy: str = 'single'  # 'single', 'multi', 'adaptive'


class QNNClassifier(HybridQuantumClassicalModel):
    """
    Quantum Neural Network Classifier.
    
    Uses variational quantum circuits to classify data into
    discrete categories. Supports multi-class classification
    via various encoding and measurement strategies.
    
    Attributes:
        num_classes: Number of output classes
        encoding_method: How to encode data into quantum state
        measurement_strategy: How to extract class predictions
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_qubits: int = 4,
        num_layers: int = 2,
        encoding_method: str = 'angle',
        measurement_strategy: str = 'single',
        name: str = "QNNClassifier"
    ):
        self.num_classes = num_classes
        self.encoding_method = encoding_method
        self.measurement_strategy = measurement_strategy
        
        super().__init__(
            input_dim=input_dim,
            output_dim=num_classes,
            num_qubits=num_qubits,
            num_quantum_layers=num_layers,
            name=name
        )
        
        # Override quantum layer for classification
        self.quantum_layer = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='strongly_entangling'
        )
        
        # Initialize class weights for multi-class
        self._initialize_class_weights()
    
    def _initialize_class_weights(self):
        """Initialize weights for multi-class output."""
        if self.num_classes > 2:
            # Use one-vs-rest style with single qubit measurement
            self.class_observables = []
            for c in range(self.num_classes):
                obs = 'Z' + 'I' * (self.num_qubits - 1)
                self.class_observables.append(self.quantum_layer._pauli_string_to_matrix(obs))
        else:
            # Binary classification: single observable
            self.class_observables = [self.quantum_layer._pauli_string_to_matrix('Z' * self.num_qubits)]
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input data into quantum circuit parameters.
        
        Args:
            x: Input vector
            
        Returns:
            Encoded parameters for quantum circuit
        """
        if self.encoding_method == 'angle':
            # Angle encoding: use input values as rotation angles
            # Normalize to [-π, π]
            x_norm = np.clip(x, -1, 1) * np.pi
            return x_norm[:self.num_qubits]
        
        elif self.encoding_method == 'amplitude':
            # Amplitude encoding: encode in quantum amplitudes
            # Requires 2^n amplitudes for n qubits
            x_norm = x / (np.linalg.norm(x) + 1e-8)
            
            # Pad or truncate to match qubit count
            size = 2 ** self.num_qubits
            if len(x_norm) < size:
                x_padded = np.zeros(size, dtype=complex)
                x_padded[:len(x_norm)] = x_norm
            else:
                x_padded = x_norm[:size]
            
            return np.abs(x_padded) ** 2  # Use probabilities
        
        elif self.encoding_method == 'basis':
            # Basis encoding: encode as computational basis state
            # For integer inputs
            idx = int(np.clip(x[0], 0, 2 ** self.num_qubits - 1))
            return np.array([float(idx)])
        
        else:
            return x[:self.num_qubits]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for classification.
        
        Args:
            x: Input data (batch_size, input_dim)
            
        Returns:
            Class probabilities (batch_size, num_classes)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_size = x.shape[0]
        
        # Encode and process through quantum circuit
        predictions = []
        
        for i in range(batch_size):
            # Encode input
            encoded = self.encode(x[i])
            
            # Execute quantum circuit
            state = self.quantum_layer.forward(encoded)
            
            # Extract class probabilities from measurement
            probs = self._measure_class(state)
            predictions.append(probs)
        
        predictions = np.array(predictions)
        
        # Apply softmax for multi-class
        if self.num_classes > 1:
            predictions = self._softmax(predictions)
        
        return predictions
    
    def _measure_class(self, state) -> np.ndarray:
        """Extract class probabilities from quantum state."""
        probs = np.abs(state.state_vector) ** 2
        
        if self.measurement_strategy == 'single':
            # Measure single qubit and interpret as class
            n = self.num_qubits
            # Probability of |0⟩ on first qubit
            prob_zero = sum(probs[i] for i in range(0, len(probs), 2))
            prob_one = sum(probs[i] for i in range(1, len(probs), 2))
            
            return np.array([prob_zero, prob_one])
        
        elif self.measurement_strategy == 'multi':
            # Measure multiple qubits
            predictions = []
            for q in range(min(self.num_classes, self.num_qubits)):
                offset = 2 ** (self.num_qubits - 1 - q)
                p = sum(probs[i] for i in range(offset, len(probs), 2 * offset))
                predictions.append(p)
            
            # Pad if needed
            while len(predictions) < self.num_classes:
                predictions.append(0.0)
            
            return np.array(predictions[:self.num_classes])
        
        else:
            # Adaptive measurement
            return probs[:self.num_classes]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax activation."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            x: Input data
            
        Returns:
            Class labels (batch_size,)
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=-1)
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            x: Input data
            
        Returns:
            Class probabilities (batch_size, num_classes)
        """
        return self.forward(x)
    
    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            y_true: True labels (batch_size,) or (batch_size, num_classes)
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        # Convert labels to one-hot if needed
        if y_true.ndim == 1:
            y_true_onehot = np.zeros_like(y_pred)
            for i, label in enumerate(y_true):
                y_true_onehot[i, int(label)] = 1.0
            y_true = y_true_onehot
        
        # Clip predictions for numerical stability
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        # Cross-entropy
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        
        return loss
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Backward pass computing gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Gradient w.r.t. input
        """
        # Cross-entropy gradient
        grad = y_pred - y_true
        
        # Scale gradient
        grad = grad / len(y_true)
        
        return grad


class QuantumKernelClassifier:
    """
    Quantum Kernel-based Classifier.
    
    Uses quantum kernels to measure similarity between data points
    in a high-dimensional Hilbert space, then applies classical
    classification.
    """
    
    def __init__(
        self,
        num_qubits: int = 4,
        kernel_type: str = 'amplitude',
        classifier: str = 'svm',  # 'svm', 'knn', 'logistic'
        C: float = 1.0
    ):
        self.num_qubits = num_qubits
        self.kernel_type = kernel_type
        self.classifier_type = classifier
        self.C = C
        
        # Kernel matrix
        self.K = None
        self.X_train = None
        self.y_train = None
    
    def _compute_kernel_vector(self, x1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel between one point and a set of points."""
        n = len(X2)
        k = np.zeros(n)
        
        for i in range(n):
            k[i] = self._compute_kernel(x1, X2[i])
        
        return k
    
    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum kernel between two points.
        
        K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²
        
        where |φ(x)⟩ is the quantum feature map of x.
        """
        # Normalize inputs
        x1_norm = x1 / (np.linalg.norm(x1) + 1e-8)
        x2_norm = x2 / (np.linalg.norm(x2) + 1e-8)
        
        if self.kernel_type == 'amplitude':
            # Amplitude-based kernel
            overlap = np.abs(np.vdot(x1_norm, x2_norm)) ** 2
            return overlap
        
        elif self.kernel_type == 'hilbert':
            # Hilbert-Schmidt inner product
            # Encode into quantum states and compute overlap
            # Simplified: use classical fidelity as proxy
            return np.exp(-np.sum((x1 - x2) ** 2) / 2)
        
        else:
            return np.dot(x1_norm, x2_norm)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the kernel classifier.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        self.X_train = X
        self.y_train = y
        n = len(X)
        
        # Compute kernel matrix
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i, j] = self._compute_kernel(X[i], X[j])
        
        if self.classifier_type == 'svm':
            self._fit_svm()
        elif self.classifier_type == 'knn':
            pass  # KNN doesn't need fitting
        elif self.classifier_type == 'logistic':
            self._fit_logistic()
    
    def _fit_svm(self):
        """Fit SVM in kernel space."""
        # Simplified SVM training using kernel matrix
        n = len(self.y_train)
        
        # Dual formulation: max α Σ α_i - 0.5 Σ α_i α_j y_i y_j K(x_i, x_j)
        # Using SMO-like approach would be needed for full implementation
        pass
    
    def _fit_logistic(self):
        """Fit logistic regression in kernel space."""
        # Kernel logistic regression
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        n_test = len(X)
        n_train = len(self.X_train)
        
        predictions = []
        
        for i in range(n_test):
            # Compute kernel with training data
            k = self._compute_kernel_vector(X[i], self.X_train)
            
            if self.classifier_type == 'knn':
                # K-Nearest Neighbors
                neighbors = np.argsort(k)[-5:]  # k=5
                labels = self.y_train[neighbors]
                pred = np.bincount(labels.astype(int)).argmax()
                predictions.append(pred)
            
            elif self.classifier_type == 'svm':
                # Kernel SVM prediction
                # y = sign(Σ α_i y_i K(x_i, x))
                pred = np.sign(np.sum(k * self.y_train))
                predictions.append(pred)
            
            else:
                # Simple weighted vote
                pred = np.sum(k * self.y_train) / (np.sum(k) + 1e-8)
                predictions.append(1 if pred > 0 else 0)
        
        return np.array(predictions)
