"""
Quantum Recurrent Neural Network Layer

Implements quantum versions of recurrent layers including
simple RNN cells, LSTM cells, and GRU cells.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
import copy

from quantumcortex.core.quantum_state import QuantumState
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit


@dataclass
class QuantumRNNConfig:
    """Configuration for quantum RNN."""
    hidden_size: int = 16
    num_qubits: int = 6
    num_layers: int = 2
    activation: str = 'tanh'
    use_bias: bool = True
    return_sequences: bool = True


class QuantumRecurrentLayer:
    """
    Quantum Recurrent Neural Network Layer.
    
    Uses quantum circuits to implement the hidden state transformation
    in recurrent networks. Each timestep processes input and previous
    hidden state through a variational quantum circuit.
    
    Attributes:
        config: Layer configuration
        input_to_hidden: PQC for input transformation
        hidden_to_hidden: PQC for hidden state transformation
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        config: Optional[QuantumRNNConfig] = None,
        name: str = "QuantumRNN"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.config = config or QuantumRNNConfig(hidden_size=hidden_size)
        
        # Combine input and hidden dimensions for quantum processing
        self.combined_size = input_size + hidden_size
        
        # Initialize quantum circuits
        self._initializeCircuits()
        
        # Classical projection weights
        self._initialize_weights()
    
    def _initializeCircuits(self):
        """Initialize quantum circuits for transformations."""
        # Input to quantum state circuit
        self.input_circuit = ParameterizedQuantumCircuit(
            num_qubits=min(self.config.num_qubits, self.input_size),
            num_layers=self.config.num_layers
        )
        
        # Hidden state circuit
        self.hidden_circuit = ParameterizedQuantumCircuit(
            num_qubits=min(self.config.num_qubits, self.hidden_size),
            num_layers=self.config.num_layers
        )
        
        # Output circuit
        self.output_circuit = ParameterizedQuantumCircuit(
            num_qubits=min(self.config.num_qubits, self.hidden_size),
            num_layers=self.config.num_layers
        )
    
    def _initialize_weights(self):
        """Initialize classical projection weights."""
        # Input to hidden projection
        self.Wxh = np.random.randn(self.input_size, self.hidden_size) * 0.01
        
        # Hidden to hidden projection
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        
        # Hidden to output projection
        self.Why = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        
        # Biases
        if self.config.use_bias:
            self.bh = np.zeros(self.hidden_size)
            self.by = np.zeros(self.hidden_size)
        else:
            self.bh = None
            self.by = None
    
    def _quantum_transform(
        self,
        x: np.ndarray,
        circuit: ParameterizedQuantumCircuit
    ) -> np.ndarray:
        """
        Apply quantum transformation to input.
        
        Args:
            x: Input vector
            circuit: Quantum circuit to use
            
        Returns:
            Transformed vector
        """
        # Encode input into quantum state
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        
        # Execute quantum circuit
        state = circuit.forward(x_norm)
        
        # Extract output from amplitudes
        probs = np.abs(state.state_vector) ** 2
        
        # Project to output dimension
        output_size = min(len(probs), self.hidden_size)
        return probs[:output_size]
    
    def forward(
        self,
        x: np.ndarray,
        hidden_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through RNN.
        
        Args:
            x: Input sequence (batch, seq_len, input_size) or (seq_len, input_size)
            hidden_state: Initial hidden state
            
        Returns:
            Tuple of (outputs, final_hidden_state)
        """
        # Handle 2D input
        if x.ndim == 2:
            x = x.unsqueeze(0) if hasattr(x, 'unsqueeze') else x.reshape(1, *x.shape)
        
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        if hidden_state is None:
            hidden_state = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        
        for t in range(seq_len):
            # Get input at timestep t
            x_t = x[:, t, :]
            
            # Compute new hidden state
            # h_new = activation(Wxh @ x_t + Whh @ h_old + bh)
            h_input = x_t @ self.Wxh + hidden_state @ self.Whh
            if self.bh is not None:
                h_input += self.bh
            
            # Apply quantum transformation
            h_quantum = self._quantum_transform(h_input, self.output_circuit)
            
            # Combine classical and quantum
            if len(h_quantum) < self.hidden_size:
                h_quantum = np.pad(h_quantum, (0, self.hidden_size - len(h_quantum)))
            
            # Apply activation
            if self.config.activation == 'tanh':
                h_new = np.tanh(h_quantum[:self.hidden_size])
            elif self.config.activation == 'relu':
                h_new = np.maximum(0, h_quantum[:self.hidden_size])
            else:
                h_new = h_quantum[:self.hidden_size]
            
            outputs.append(h_new)
            hidden_state = h_new
        
        # Stack outputs
        outputs = np.stack(outputs, axis=1)  # (batch, seq_len, hidden_size)
        
        if not self.config.return_sequences:
            outputs = outputs[:, -1, :]  # Return only last output
        
        return outputs, hidden_state
    
    def __call__(
        self,
        x: np.ndarray,
        hidden_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x, hidden_state)


class QuantumLSTMCell:
    """
    Quantum LSTM Cell.
    
    Implements Long Short-Term Memory gates using quantum circuits
    for the gate computations.
    
    Gates:
    - Forget gate: what to forget from cell state
    - Input gate: what to add to cell state
    - Output gate: what to output
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_qubits: int = 8,
        name: str = "QuantumLSTMCell"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits
        self.name = name
        
        # Gate computation circuits
        self.forget_circuit = self._build_gate_circuit()
        self.input_circuit = self._build_gate_circuit()
        self.output_circuit = self._build_gate_circuit()
        self.cell_circuit = self._build_gate_circuit()
        
        # Classical projections
        self._initialize_weights()
    
    def _build_gate_circuit(self) -> ParameterizedQuantumCircuit:
        """Build a circuit for computing a gate."""
        circuit = ParameterizedQuantumCircuit(
            num_qubits=self.num_qubits,
            num_layers=1
        )
        circuit.add_layer('single_layer', 'gate')
        return circuit
    
    def _initialize_weights(self):
        """Initialize weight matrices and biases."""
        combined_size = self.input_size + self.hidden_size
        
        # Input concatenated weights for all gates
        # Each gate: combined_size -> hidden_size
        self.Wf = np.random.randn(combined_size, self.hidden_size) * 0.01
        self.Wi = np.random.randn(combined_size, self.hidden_size) * 0.01
        self.Wc = np.random.randn(combined_size, self.hidden_size) * 0.01
        self.Wo = np.random.randn(combined_size, self.hidden_size) * 0.01
        
        # Biases
        self.bf = np.zeros(self.hidden_size)
        self.bi = np.zeros(self.hidden_size)
        self.bc = np.zeros(self.hidden_size)
        self.bo = np.zeros(self.hidden_size)
    
    def _quantum_gate(
        self,
        x: np.ndarray,
        circuit: ParameterizedQuantumCircuit
    ) -> np.ndarray:
        """Apply quantum gate computation."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        state = circuit.forward(x_norm)
        probs = np.abs(state.state_vector) ** 2
        return probs[:self.hidden_size]
    
    def forward(
        self,
        x_t: np.ndarray,
        hidden_state: np.ndarray,
        cell_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for single timestep.
        
        Args:
            x_t: Input at timestep t (batch, input_size)
            hidden_state: Previous hidden state (batch, hidden_size)
            cell_state: Previous cell state (batch, hidden_size)
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        # Concatenate input and hidden
        combined = np.concatenate([x_t, hidden_state], axis=1)
        
        # Forget gate
        f = combined @ self.Wf + self.bf
        f_quantum = self._quantum_gate(f, self.forget_circuit)
        f = torch_sigmoid(np.clip(f_quantum, -10, 10))
        
        # Input gate
        i = combined @ self.Wi + self.bi
        i_quantum = self._quantum_gate(i, self.input_circuit)
        i = torch_sigmoid(np.clip(i_quantum, -10, 10))
        
        # Cell candidate
        c_candidate = combined @ self.Wc + self.bc
        c_quantum = self._quantum_gate(c_candidate, self.cell_circuit)
        c_tilde = np.tanh(np.clip(c_quantum, -10, 10))
        
        # Output gate
        o = combined @ self.Wo + self.bo
        o_quantum = self._quantum_gate(o, self.output_circuit)
        o = torch_sigmoid(np.clip(o_quantum, -10, 10))
        
        # New cell state
        new_cell = f * cell_state + i * c_tilde
        
        # New hidden state
        new_hidden = o * np.tanh(new_cell)
        
        return new_hidden, new_cell
    
    def __call__(
        self,
        x_t: np.ndarray,
        hidden_state: np.ndarray,
        cell_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x_t, hidden_state, cell_state)


def torch_sigmoid(x):
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class QuantumLSTMLayer:
    """
    Full Quantum LSTM Layer with sequence processing.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_qubits: int = 8,
        name: str = "QuantumLSTM"
    ):
        self.lstm_cell = QuantumLSTMCell(input_size, hidden_size, num_qubits, name)
        self.hidden_size = hidden_size
    
    def forward(
        self,
        x: np.ndarray,
        hidden_state: Optional[np.ndarray] = None,
        cell_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through LSTM layer.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            hidden_state: Initial hidden state
            cell_state: Initial cell state
            
        Returns:
            Tuple of (outputs, (final_hidden, final_cell))
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize states
        if hidden_state is None:
            hidden_state = np.zeros((batch_size, self.hidden_size))
        if cell_state is None:
            cell_state = np.zeros((batch_size, self.hidden_size))
        
        outputs = []
        
        for t in range(seq_len):
            hidden_state, cell_state = self.lstm_cell(
                x[:, t, :],
                hidden_state,
                cell_state
            )
            outputs.append(hidden_state)
        
        outputs = np.stack(outputs, axis=1)
        
        return outputs, (hidden_state, cell_state)
    
    def __call__(
        self,
        x: np.ndarray,
        hidden_state: Optional[np.ndarray] = None,
        cell_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.forward(x, hidden_state, cell_state)


class QuantumGRUCell:
    """
    Quantum Gated Recurrent Unit (GRU) Cell.
    
    Lighter than LSTM with fewer gates:
    - Update gate: how much past to keep
    - Reset gate: how much past to forget
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_qubits: int = 6,
        name: str = "QuantumGRUCell"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_qubits = num_qubits
        self.name = name
        
        # Gate circuits
        self.z_gate_circuit = self._build_gate_circuit()
        self.r_gate_circuit = self._build_gate_circuit()
        self.h_circuit = self._build_gate_circuit()
        
        self._initialize_weights()
    
    def _build_gate_circuit(self) -> ParameterizedQuantumCircuit:
        """Build gate computation circuit."""
        circuit = ParameterizedQuantumCircuit(
            num_qubits=self.num_qubits,
            num_layers=1
        )
        circuit.add_layer('single_layer', 'gru_gate')
        return circuit
    
    def _initialize_weights(self):
        """Initialize GRU weights."""
        combined = self.input_size + self.hidden_size
        
        # Update gate
        self.Wz = np.random.randn(combined, self.hidden_size) * 0.01
        self.bz = np.zeros(self.hidden_size)
        
        # Reset gate
        self.Wr = np.random.randn(combined, self.hidden_size) * 0.01
        self.br = np.zeros(self.hidden_size)
        
        # Hidden candidate
        self.Wh = np.random.randn(combined, self.hidden_size) * 0.01
        self.bh = np.zeros(self.hidden_size)
    
    def _quantum_gate(self, x: np.ndarray, circuit) -> np.ndarray:
        """Quantum gate computation."""
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        state = circuit.forward(x_norm)
        probs = np.abs(state.state_vector) ** 2
        return probs[:self.hidden_size]
    
    def forward(
        self,
        x_t: np.ndarray,
        hidden_state: np.ndarray
    ) -> np.ndarray:
        """Forward pass for single timestep."""
        combined = np.concatenate([x_t, hidden_state], axis=1)
        
        # Update gate
        z = combined @ self.Wz + self.bz
        z_quantum = self._quantum_gate(z, self.z_gate_circuit)
        z = torch_sigmoid(np.clip(z_quantum, -10, 10))
        
        # Reset gate
        r = combined @ self.Wr + self.br
        r_quantum = self._quantum_gate(r, self.r_gate_circuit)
        r = torch_sigmoid(np.clip(r_quantum, -10, 10))
        
        # Hidden candidate
        combined_reset = np.concatenate([x_t, r * hidden_state], axis=1)
        h_tilde = combined_reset @ self.Wh + self.bh
        h_quantum = self._quantum_gate(h_tilde, self.h_circuit)
        h_tilde = np.tanh(np.clip(h_quantum, -10, 10))
        
        # New hidden state
        new_hidden = (1 - z) * hidden_state + z * h_tilde
        
        return new_hidden
    
    def __call__(self, x_t: np.ndarray, hidden_state: np.ndarray) -> np.ndarray:
        return self.forward(x_t, hidden_state)


class QuantumBidirectionalRNN:
    """
    Bidirectional Quantum RNN.
    
    Processes sequence in both forward and backward directions,
    concatenating outputs for richer representation.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_qubits: int = 6,
        name: str = "QuantumBiRNN"
    ):
        self.forward_rnn = QuantumRecurrentLayer(
            input_size, hidden_size,
            QuantumRNNConfig(num_qubits=num_qubits),
            name=f"{name}_forward"
        )
        
        self.backward_rnn = QuantumRecurrentLayer(
            input_size, hidden_size,
            QuantumRNNConfig(num_qubits=num_qubits),
            name=f"{name}_backward"
        )
        
        self.hidden_size = hidden_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with bidirectional processing.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            
        Returns:
            Concatenated forward/backward outputs (batch, seq_len, 2*hidden_size)
        """
        # Forward pass
        out_forward, _ = self.forward_rnn(x)
        
        # Backward pass (reverse sequence)
        x_reversed = np.flip(x, axis=1)
        out_backward, _ = self.backward_rnn(x_reversed)
        out_backward = np.flip(out_backward, axis=1)
        
        # Concatenate
        return np.concatenate([out_forward, out_backward], axis=-1)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantumAttentionRNN:
    """
    Quantum RNN with Attention mechanism.
    
    Adds attention over past hidden states for better
    context modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        attention_size: int = 16,
        num_qubits: int = 6,
        name: str = "QuantumAttentionRNN"
    ):
        self.rnn = QuantumRecurrentLayer(
            input_size, hidden_size,
            QuantumRNNConfig(num_qubits=num_qubits)
        )
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Attention weights
        self.Wa = np.random.randn(hidden_size, attention_size) * 0.01
        self.Ua = np.random.randn(hidden_size, attention_size) * 0.01
        self.Va = np.random.randn(attention_size, 1) * 0.01
    
    def forward(
        self,
        x: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with attention.
        
        Args:
            x: Input sequence (batch, seq_len, input_size)
            context: Optional external context
            
        Returns:
            Tuple of (attended_outputs, attention_weights)
        """
        # Get RNN outputs
        outputs, final_hidden = self.rnn(x)
        
        # Compute attention
        batch_size, seq_len, hidden = outputs.shape
        
        # Expanded attention scores
        # e_t = v^T tanh(W_a * h_t + U_a * h_s)
        # where h_s is a summary context (last hidden or external)
        
        if context is None:
            context = outputs[:, -1, :]  # Use last hidden state as context
        
        # Expand context to all timesteps
        context_expanded = np.repeat(context[:, np.newaxis, :], seq_len, axis=1)
        
        # Attention scores
        e = np.tanh(
            outputs @ self.Wa + context_expanded @ self.Ua
        ) @ self.Va
        
        # Softmax
        e_exp = np.exp(e - np.max(e, axis=1, keepdims=True))
        attention_weights = e_exp / (np.sum(e_exp, axis=1, keepdims=True) + 1e-8)
        
        # Context vector
        context_vector = np.sum(attention_weights * outputs, axis=1)
        
        return context_vector, attention_weights
    
    def __call__(
        self,
        x: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(x, context)
