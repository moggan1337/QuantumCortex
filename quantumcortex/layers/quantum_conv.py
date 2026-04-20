"""
Quantum Convolutional Neural Network Layer

Implements quantum versions of convolutional layers for
processing structured data like images using quantum circuits.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
import copy

from quantumcortex.core.quantum_state import QuantumState, QuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit


@dataclass
class QuantumConvConfig:
    """Configuration for quantum convolutional layer."""
    kernel_size: Tuple[int, int] = (3, 3)
    filters: int = 8
    stride: Tuple[int, int] = (1, 1)
    padding: str = 'valid'  # 'valid', 'same', or (pad_h, pad_w)
    num_qubits_per_patch: int = 9  # 3x3 = 9 pixels
    num_layers: int = 2
    activation: Optional[str] = 'relu'


class QuantumConvolutionalLayer:
    """
    Quantum Convolutional Layer.
    
    Applies quantum circuits to local patches of input data,
    mimicking the operation of classical convolutional layers.
    
    The layer slides a quantum kernel (parameterized quantum circuit)
    across the input, processing each patch independently and
    combining the results.
    
    Attributes:
        config: Layer configuration
        kernels: List of parameterized quantum circuits (one per filter)
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        config: Optional[QuantumConvConfig] = None,
        name: str = "QuantumConv2D"
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.name = name
        self.config = config or QuantumConvConfig(filters=output_channels)
        
        # Override config with output_channels
        self.config.filters = output_channels
        
        # Initialize quantum kernels (one per output filter)
        self.kernels: List[ParameterizedQuantumCircuit] = []
        self._initialize_kernels()
        
        # Classical preprocessing weights
        self._initialize_weights()
    
    def _initialize_kernels(self):
        """Initialize quantum kernels for each filter."""
        for _ in range(self.config.filters):
            kernel = ParameterizedQuantumCircuit(
                num_qubits=self.config.num_qubits_per_patch,
                num_layers=self.config.num_layers
            )
            
            # Add kernel structure
            for layer_idx in range(self.config.num_layers):
                kernel.add_layer('single_layer', f'kernel_layer_{layer_idx}')
            
            self.kernels.append(kernel)
    
    def _initialize_weights(self):
        """Initialize classical preprocessing weights."""
        patch_size = self.config.kernel_size[0] * self.config.kernel_size[1]
        self.input_weights = np.random.randn(
            self.input_channels,
            patch_size,
            self.config.num_qubits_per_patch
        ) * 0.01
        
        self.biases = [np.zeros(self.config.num_qubits_per_patch) 
                      for _ in range(self.config.filters)]
    
    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Apply padding to input."""
        if self.config.padding == 'valid':
            return x
        
        if self.config.padding == 'same':
            pad_h = self.config.kernel_size[0] // 2
            pad_w = self.config.kernel_size[1] // 2
        elif isinstance(self.config.padding, tuple):
            pad_h, pad_w = self.config.padding
        else:
            pad_h, pad_w = 0, 0
        
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    
    def _extract_patches(
        self,
        x: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract patches from input.
        
        Args:
            x: Input tensor (batch, height, width, channels)
            
        Returns:
            List of (patch, position) tuples
        """
        patches = []
        stride_h, stride_w = self.config.stride
        kernel_h, kernel_w = self.config.kernel_size
        
        batch_size, height, width, _ = x.shape
        
        for b in range(batch_size):
            for i in range(0, height - kernel_h + 1, stride_h):
                for j in range(0, width - kernel_w + 1, stride_w):
                    patch = x[b, i:i+kernel_h, j:j+kernel_w, :]
                    patches.append((patch, (b, i, j)))
        
        return patches
    
    def _encode_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Encode a patch into quantum parameters.
        
        Args:
            patch: (kernel_h, kernel_w, input_channels)
            
        Returns:
            Parameters for quantum circuit
        """
        # Flatten and preprocess patch
        patch_flat = patch.flatten()  # (kernel_h * kernel_w * channels,)
        
        # Apply input weights
        encoded = patch_flat @ self.input_weights.reshape(
            -1, self.config.num_qubits_per_patch
        )
        
        # Normalize to valid rotation range
        encoded = np.clip(encoded, -np.pi, np.pi)
        
        return encoded
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through quantum convolutional layer.
        
        Args:
            x: Input tensor (batch, height, width, channels)
            
        Returns:
            Output tensor (batch, out_height, out_width, filters)
        """
        # Pad input if needed
        x_padded = self._pad_input(x)
        
        # Extract patches
        patches = self._extract_patches(x_padded)
        
        # Determine output dimensions
        stride_h, stride_w = self.config.stride
        kernel_h, kernel_w = self.config.kernel_size
        batch_size, height, width, _ = x.shape
        
        out_h = (height + 2 * (self.config.padding[0] if isinstance(self.config.padding, tuple) 
                               else (kernel_h // 2 if self.config.padding == 'same' else 0)) - kernel_h) // stride_h + 1
        out_w = (width + 2 * (self.config.padding[1] if isinstance(self.config.padding, tuple) 
                              else (kernel_w // 2 if self.config.padding == 'same' else 0)) - kernel_w) // stride_w + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_h, out_w, self.config.filters))
        
        # Process patches
        patch_idx = 0
        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    patch, _ = patches[patch_idx]
                    patch_idx += 1
                    
                    # Encode patch
                    encoded = self._encode_patch(patch)
                    
                    # Apply each quantum kernel
                    for f in range(self.config.filters):
                        # Execute quantum kernel
                        state = self.kernels[f].forward(encoded)
                        
                        # Extract output from measurement
                        probs = np.abs(state.state_vector) ** 2
                        output_val = np.mean(probs[:self.config.num_qubits_per_patch])
                        
                        # Add bias and activation
                        output_val += self.biases[f].mean()
                        if self.config.activation == 'relu':
                            output_val = max(0, output_val)
                        
                        output[b, i, j, f] = output_val
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantumPoolingLayer:
    """
    Quantum Pooling Layer.
    
    Implements pooling operations (max, average) using
    quantum measurement and interference.
    """
    
    def __init__(
        self,
        pool_size: Tuple[int, int] = (2, 2),
        stride: Tuple[int, int] = (2, 2),
        pool_type: str = 'max',
        name: str = "QuantumPool"
    ):
        self.pool_size = pool_size
        self.stride = stride
        self.pool_type = pool_type
        self.name = name
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through pooling layer.
        
        Args:
            x: Input tensor (batch, height, width, channels)
            
        Returns:
            Output tensor
        """
        batch_size, height, width, channels = x.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        
        out_h = (height - pool_h) // stride_h + 1
        out_w = (width - pool_w) // stride_w + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        
                        patch = x[b, h_start:h_start+pool_h, 
                                  w_start:w_start+pool_w, c]
                        
                        if self.pool_type == 'max':
                            output[b, i, j, c] = np.max(patch)
                        elif self.pool_type == 'average':
                            output[b, i, j, c] = np.mean(patch)
                        elif self.pool_type == 'quantum':
                            # Quantum pooling: use quantum interference
                            output[b, i, j, c] = self._quantum_pool(patch)
        
        return output
    
    def _quantum_pool(self, patch: np.ndarray) -> float:
        """
        Quantum pooling operation.
        
        Uses quantum amplitude estimation-like approach
        to combine patch values.
        """
        # Simple quantum-inspired pooling
        values = patch.flatten()
        
        # Amplitude-based combination
        amplitudes = np.abs(values)
        amplitudes = amplitudes / (np.sum(amplitudes) + 1e-8)
        
        # Use interference-like combination
        phases = np.angle(values)
        
        # Combined result using weighted amplitudes
        result = np.sum(amplitudes * values)
        
        return np.real(result)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Quantum1DConvolution:
    """
    Quantum 1D Convolutional Layer.
    
    For processing sequential data like time series or text.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        num_qubits_per_kernel: int = 4,
        num_layers: int = 2,
        stride: int = 1,
        name: str = "QuantumConv1D"
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_qubits = num_qubits_per_kernel
        self.num_layers = num_layers
        self.stride = stride
        self.name = name
        
        # Initialize kernels
        self.kernels: List[ParameterizedQuantumCircuit] = []
        self._initialize_kernels()
    
    def _initialize_kernels(self):
        """Initialize quantum kernels."""
        for _ in range(self.output_channels):
            kernel = ParameterizedQuantumCircuit(
                num_qubits=self.num_qubits,
                num_layers=self.num_layers
            )
            kernel.add_layer('single_layer', 'conv_kernel')
            self.kernels.append(kernel)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input (batch, sequence, channels)
            
        Returns:
            Output (batch, out_sequence, output_channels)
        """
        batch_size, seq_len, channels = x.shape
        
        out_len = (seq_len - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_len, self.output_channels))
        
        for b in range(batch_size):
            for i in range(out_len):
                start = i * self.stride
                patch = x[b, start:start + self.kernel_size, :]
                
                for f in range(self.output_channels):
                    # Encode patch
                    patch_flat = patch.flatten()
                    params = patch_flat[:self.num_qubits]
                    
                    # Execute kernel
                    state = self.kernels[f].forward(params)
                    probs = np.abs(state.state_vector) ** 2
                    
                    output[b, i, f] = np.mean(probs)
        
        return output
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantumSeparableConv2D:
    """
    Quantum Separable Convolution.
    
    Depthwise convolution followed by pointwise convolution,
    with quantum circuits for both stages.
    """
    
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        name: str = "QuantumDepthwiseSeparable"
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution
        self.depthwise = QuantumConvolutionalLayer(
            input_channels=input_channels,
            output_channels=input_channels,  # Same as input for depthwise
            config=QuantumConvConfig(
                kernel_size=kernel_size,
                filters=input_channels
            )
        )
        
        # Pointwise convolution
        self.pointwise = Quantum1DConvolution(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=1
        )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Depthwise step
        x = self.depthwise(x)
        
        # Pointwise step (reshape for 1D conv)
        batch, h, w, c = x.shape
        x_reshaped = x.reshape(batch, h * w, c)
        
        # Apply pointwise
        out = self.pointwise(x_reshaped)
        
        # Reshape back
        out_len = out.shape[1]
        out_h = h
        out_w = out_len // out_h
        
        return out.reshape(batch, out_h, out_w, self.output_channels)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
