"""
Quantum State and Circuit Module

Implements quantum state representations and circuit operations
for the QuantumCortex framework.
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
import copy


# Pauli matrices
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Phase gates
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Swap and controlled gates
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)
CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)
CPHASE = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=complex)


def kron(*matrices: np.ndarray) -> np.array:
    """Kronecker product of multiple matrices."""
    result = np.array([[1]], dtype=complex)
    for m in matrices:
        result = np.kron(result, m)
    return result


def controlled_unitary(U: np.ndarray) -> np.ndarray:
    """Create controlled version of a unitary gate."""
    n = U.shape[0]
    controlled = np.zeros((2 * n, 2 * n), dtype=complex)
    controlled[:n, :n] = np.eye(n)
    controlled[n:, n:] = U
    return controlled


@dataclass
class QuantumState:
    """
    Represents a quantum state vector.
    
    Attributes:
        state_vector: The complex amplitude vector representing the quantum state
        num_qubits: Number of qubits in the system
        name: Optional name for the state
    """
    state_vector: np.ndarray
    num_qubits: int
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize the quantum state."""
        if self.state_vector.shape != (2 ** self.num_qubits,):
            raise ValueError(
                f"State vector shape {self.state_vector.shape} does not match "
                f"2^{self.num_qubits} for {self.num_qubits} qubits"
            )
        # Normalize if not already normalized
        norm = np.linalg.norm(self.state_vector)
        if not np.isclose(norm, 1.0, atol=1e-10):
            self.state_vector = self.state_vector / norm
    
    @classmethod
    def zero(cls, num_qubits: int, name: Optional[str] = None) -> 'QuantumState':
        """Create the |0...0⟩ state."""
        state_vector = np.zeros(2 ** num_qubits, dtype=complex)
        state_vector[0] = 1.0
        return cls(state_vector, num_qubits, name)
    
    @classmethod
    def plus(cls, num_qubits: int, name: Optional[str] = None) -> 'QuantumState':
        """Create a uniform superposition state."""
        state_vector = np.ones(2 ** num_qubits, dtype=complex) / np.sqrt(2 ** num_qubits)
        return cls(state_vector, num_qubits, name)
    
    @classmethod
    def random(cls, num_qubits: int, name: Optional[str] = None, seed: Optional[int] = None) -> 'QuantumState':
        """Create a random quantum state."""
        if seed is not None:
            np.random.seed(seed)
        state_vector = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
        return cls(state_vector, num_qubits, name)
    
    def apply_gate(self, gate: np.ndarray, target_qubits: List[int]) -> 'QuantumState':
        """
        Apply a gate to specified target qubits.
        
        Args:
            gate: Single-qubit or multi-qubit unitary gate
            target_qubits: List of qubit indices to apply the gate to
            
        Returns:
            New QuantumState with the gate applied
        """
        num_qubits = len(target_qubits)
        gate_size = gate.shape[0]
        
        if gate_size != 2 ** num_qubits:
            raise ValueError(f"Gate size {gate_size} doesn't match {2**num_qubits} for {num_qubits} qubits")
        
        # Build the full circuit operator using Kronecker products
        full_op = self._build_gate_operator(gate, target_qubits)
        
        new_state_vector = full_op @ self.state_vector
        return QuantumState(new_state_vector, self.num_qubits, f"{self.name}_applied")
    
    def _build_gate_operator(self, gate: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Build the full N-qubit operator with the gate on target qubits."""
        target_set = set(target_qubits)
        num_target = len(target_qubits)
        
        operators = []
        for i in range(self.num_qubits - 1, -1, -1):  # Reverse order for proper Kronecker
            if i in target_set:
                idx = target_qubits.index(i)
                if num_target == 1:
                    operators.append(gate)
                else:
                    # This is a multi-qubit gate
                    operators.append(gate)
                    break
            else:
                operators.append(PAULI_I)
        
        # For single qubit gates on specific targets
        if num_target == 1:
            operators = []
            for i in range(self.num_qubits - 1, -1, -1):
                if i == target_qubits[0]:
                    operators.append(gate)
                else:
                    operators.append(PAULI_I)
        
        return kron(*operators)
    
    def apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> 'QuantumState':
        """Apply a two-qubit gate to specified qubits."""
        if self.num_qubits < 2:
            raise ValueError("Need at least 2 qubits for two-qubit gate")
        
        # Build the full operator
        full_op = self._build_two_qubit_operator(gate, qubit1, qubit2)
        new_state_vector = full_op @ self.state_vector
        return QuantumState(new_state_vector, self.num_qubits)
    
    def _build_two_qubit_operator(self, gate: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """Build full operator for two-qubit gate."""
        operators = []
        for i in range(self.num_qubits - 1, -1, -1):
            if i == q1 or i == q2:
                if len(operators) == 0 or (operators[-1] is not PAULI_I):
                    operators.append(gate)
                    break
            else:
                operators.append(PAULI_I)
        
        # Rebuild properly for two-qubit case
        operators = []
        for i in range(self.num_qubits - 1, -1, -1):
            if i == q1:
                operators.append(np.array([[1, 0], [0, 0]]))  # |0><0| part placeholder
            elif i == q2:
                operators.append(np.array([[0, 0], [0, 1]]))  # |1><1| part placeholder
            else:
                operators.append(PAULI_I)
        
        # Simplified: direct Kronecker for the two-qubit gate positions
        left_ops = []
        right_ops = []
        
        for i in range(self.num_qubits):
            if i < min(q1, q2):
                left_ops.append(PAULI_I)
            elif i > max(q1, q2):
                right_ops.append(PAULI_I)
        
        # Build two-qubit operator
        if q1 < q2:
            left = kron(*left_ops) if left_ops else np.array([[1]], dtype=complex)
            right = kron(*right_ops) if right_ops else np.array([[1]], dtype=complex)
            return kron(left, gate, right)
        else:
            left = kron(*left_ops) if left_ops else np.array([[1]], dtype=complex)
            right = kron(*right_ops) if right_ops else np.array([[1]], dtype=complex)
            return kron(left, gate, right)
    
    def measure(self, shots: int = 1000) -> dict:
        """
        Measure the quantum state in the computational basis.
        
        Args:
            shots: Number of measurement samples
            
        Returns:
            Dictionary mapping basis states to their frequencies
        """
        probs = np.abs(self.state_vector) ** 2
        states = [format(i, f'0{self.num_qubits}b') for i in range(len(probs))]
        
        measurements = np.random.choice(states, size=shots, p=probs)
        counts = {}
        for state in set(measurements):
            counts[state] = np.sum(measurements == state) / shots
        
        return counts
    
    def measure_expectation(self, observable: np.ndarray) -> complex:
        """
        Calculate expectation value of an observable.
        
        Args:
            observable: Hermitian operator (e.g., Pauli string)
            
        Returns:
            Expectation value <ψ|O|ψ⟩
        """
        return np.vdot(self.state_vector, observable @ self.state_vector)
    
    def partial_trace(self, trace_qubits: List[int]) -> np.ndarray:
        """
        Compute the reduced density matrix by tracing out specified qubits.
        
        Args:
            trace_qubits: Indices of qubits to trace out
            
        Returns:
            Reduced density matrix of remaining qubits
        """
        # Convert state vector to density matrix
        density_matrix = np.outer(self.state_vector, np.conj(self.state_vector))
        
        # Reshape for partial trace
        keep_qubits = [i for i in range(self.num_qubits) if i not in trace_qubits]
        n_keep = len(keep_qubits)
        n_trace = len(trace_qubits)
        
        # Reshape to (2^n_keep, 2^n_trace, 2^n_keep, 2^n_trace)
        shape = tuple(2 for _ in range(self.num_qubits * 2))
        reshaped = density_matrix.reshape(*([2] * (self.num_qubits * 2)))
        
        # Transpose to move traced qubits to the end
        perm = keep_qubits + trace_qubits
        perm_inv = [perm.index(i) for i in range(self.num_qubits)]
        perm_inv_rest = perm_inv[:n_keep] + perm_inv[n_keep:]
        reshaped = np.transpose(reshaped, perm_inv_rest)
        
        # Trace over the traced qubit dimensions
        reduced = reshaped
        for _ in range(n_trace):
            reduced = np.trace(reduced, axis1=n_keep-1, axis2=n_keep)
        
        return reduced
    
    def purity(self, trace_qubits: Optional[List[int]] = None) -> float:
        """
        Calculate purity of the state (or reduced state).
        
        Args:
            trace_qubits: Optional qubits to trace out first
            
        Returns:
            Purity Tr(ρ²)
        """
        if trace_qubits:
            rho = self.partial_trace(trace_qubits)
        else:
            rho = np.outer(self.state_vector, np.conj(self.state_vector))
        
        return np.real(np.trace(rho @ rho))
    
    def entanglement_entropy(self, partition: List[int]) -> float:
        """
        Calculate entanglement entropy using Schmidt decomposition.
        
        Args:
            partition: List of qubit indices for one partition
            
        Returns:
            Von Neumann entropy of the reduced density matrix
        """
        rho = self.partial_trace(partition)
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter zero eigenvalues
        
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return entropy
    
    def von_neumann_entropy(self) -> float:
        """Calculate von Neumann entropy of the full state."""
        rho = np.outer(self.state_vector, np.conj(self.state_vector))
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    def fidelity(self, other: 'QuantumState') -> float:
        """
        Calculate fidelity between two quantum states.
        
        Args:
            other: Another QuantumState
            
        Returns:
            Fidelity F(ρ, σ) = |<ψ|φ⟩|²
        """
        if self.num_qubits != other.num_qubits:
            raise ValueError("States must have same number of qubits")
        
        overlap = np.abs(np.vdot(self.state_vector, other.state_vector)) ** 2
        return np.real(overlap)
    
    def copy(self) -> 'QuantumState':
        """Create a deep copy of the quantum state."""
        return QuantumState(
            self.state_vector.copy(),
            self.num_qubits,
            self.name
        )
    
    def __repr__(self) -> str:
        return f"QuantumState(num_qubits={self.num_qubits}, name={self.name})"
    
    def __str__(self) -> str:
        """Pretty print the quantum state."""
        lines = [f"Quantum State ({self.num_qubits} qubits)"]
        if self.name:
            lines.append(f"Name: {self.name}")
        
        lines.append("\nState Vector:")
        for i, amp in enumerate(self.state_vector):
            if np.abs(amp) > 1e-10:
                state = format(i, f'0{self.num_qubits}b')
                prob = np.abs(amp) ** 2
                phase = np.angle(amp) if np.abs(amp) > 1e-10 else 0
                lines.append(f"  |{state}⟩: {amp:.6f} (p={prob:.4f}, θ={phase:.4f})")
        
        return "\n".join(lines)


@dataclass
class QuantumCircuit:
    """
    Represents a quantum circuit with a sequence of gates.
    
    Attributes:
        num_qubits: Number of qubits in the circuit
        gates: List of gate operations to apply
        measurements: List of measurement specifications
    """
    num_qubits: int
    gates: List[Tuple[str, List[int], Optional[np.ndarray]]] = field(default_factory=list)
    name: str = "circuit"
    
    def __post_init__(self):
        self.gates = []
        self._gate_queue = []
    
    def h(self, qubit: int, name: str = "H") -> 'QuantumCircuit':
        """Apply Hadamard gate."""
        self.gates.append(("h", [qubit], HADAMARD, name))
        return self
    
    def x(self, qubit: int, name: str = "X") -> 'QuantumCircuit':
        """Apply Pauli-X (NOT) gate."""
        self.gates.append(("x", [qubit], PAULI_X, name))
        return self
    
    def y(self, qubit: int, name: str = "Y") -> 'QuantumCircuit':
        """Apply Pauli-Y gate."""
        self.gates.append(("y", [qubit], PAULI_Y, name))
        return self
    
    def z(self, qubit: int, name: str = "Z") -> 'QuantumCircuit':
        """Apply Pauli-Z gate."""
        self.gates.append(("z", [qubit], PAULI_Z, name))
        return self
    
    def s(self, qubit: int, name: str = "S") -> 'QuantumCircuit':
        """Apply S (phase) gate."""
        self.gates.append(("s", [qubit], S_GATE, name))
        return self
    
    def t(self, qubit: int, name: str = "T") -> 'QuantumCircuit':
        """Apply T gate."""
        self.gates.append(("t", [qubit], T_GATE, name))
        return self
    
    def rx(self, qubit: int, theta: float, name: Optional[str] = None) -> 'QuantumCircuit':
        """Apply rotation around X axis."""
        name = name or f"Rx({theta:.3f})"
        gate = self._rotation_gate('x', theta)
        self.gates.append(("rx", [qubit], gate, name))
        return self
    
    def ry(self, qubit: int, theta: float, name: Optional[str] = None) -> 'QuantumCircuit':
        """Apply rotation around Y axis."""
        name = name or f"Ry({theta:.3f})"
        gate = self._rotation_gate('y', theta)
        self.gates.append(("ry", [qubit], gate, name))
        return self
    
    def rz(self, qubit: int, theta: float, name: Optional[str] = None) -> 'QuantumCircuit':
        """Apply rotation around Z axis."""
        name = name or f"Rz({theta:.3f})"
        gate = self._rotation_gate('z', theta)
        self.gates.append(("rz", [qubit], gate, name))
        return self
    
    def u3(self, qubit: int, theta: float, phi: float, lam: float, name: Optional[str] = None) -> 'QuantumCircuit':
        """Apply universal U3 gate."""
        name = name or f"U3({theta:.3f},{phi:.3f},{lam:.3f})"
        gate = self._u3_gate(theta, phi, lam)
        self.gates.append(("u3", [qubit], gate, name))
        return self
    
    def cnot(self, control: int, target: int, name: str = "CNOT") -> 'QuantumCircuit':
        """Apply CNOT gate."""
        self.gates.append(("cnot", [control, target], CNOT, name))
        return self
    
    def cz(self, qubit1: int, qubit2: int, name: str = "CZ") -> 'QuantumCircuit':
        """Apply CZ gate."""
        self.gates.append(("cz", [qubit1, qubit2], CZ, name))
        return self
    
    def swap(self, qubit1: int, qubit2: int, name: str = "SWAP") -> 'QuantumCircuit':
        """Apply SWAP gate."""
        self.gates.append(("swap", [qubit1, qubit2], SWAP, name))
        return self
    
    def controlled_u(self, control: int, target: int, U: np.ndarray, name: Optional[str] = None) -> 'QuantumCircuit':
        """Apply controlled-U gate."""
        name = name or "CU"
        cu = controlled_unitary(U)
        self.gates.append(("cu", [control, target], cu, name))
        return self
    
    def reset(self, qubit: int, state: int = 0) -> 'QuantumCircuit':
        """Reset qubit to |0⟩ or |1⟩."""
        self.gates.append(("reset", [qubit], np.array([state]), f"Reset({state})"))
        return self
    
    @staticmethod
    def _rotation_gate(axis: str, theta: float) -> np.ndarray:
        """Generate rotation gate matrix."""
        cos_t2 = np.cos(theta / 2)
        sin_t2 = np.sin(theta / 2)
        
        if axis == 'x':
            return np.array([[cos_t2, -1j * sin_t2], [-1j * sin_t2, cos_t2]], dtype=complex)
        elif axis == 'y':
            return np.array([[cos_t2, -sin_t2], [sin_t2, cos_t2]], dtype=complex)
        elif axis == 'z':
            return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
        else:
            raise ValueError(f"Unknown axis: {axis}")
    
    @staticmethod
    def _u3_gate(theta: float, phi: float, lam: float) -> np.ndarray:
        """Generate universal U3 gate."""
        return np.array([
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
        ], dtype=complex)
    
    def execute(self, initial_state: Optional[QuantumState] = None) -> QuantumState:
        """
        Execute the circuit and return the final quantum state.
        
        Args:
            initial_state: Optional initial state (defaults to |0...0⟩)
            
        Returns:
            Final QuantumState after applying all gates
        """
        if initial_state is None:
            state = QuantumState.zero(self.num_qubits)
        else:
            state = initial_state.copy()
        
        for gate_type, qubits, matrix, _ in self.gates:
            if gate_type in ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u3"]:
                state = state.apply_gate(matrix, qubits)
            elif gate_type in ["cnot", "cz", "swap", "cu"]:
                state = state.apply_two_qubit_gate(matrix, qubits[0], qubits[1])
            elif gate_type == "reset":
                # Handle reset operation
                pass
        
        return state
    
    def get_unitary(self) -> np.ndarray:
        """
        Compute the unitary matrix representing the circuit.
        
        Returns:
            Unitary matrix of shape (2^n, 2^n)
        """
        U = np.eye(2 ** self.num_qubits, dtype=complex)
        
        for gate_type, qubits, matrix, _ in self.gates:
            if gate_type in ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "u3"]:
                # Build single-qubit operator
                op = self._build_single_qubit_op(matrix, qubits[0])
                U = op @ U
            elif gate_type in ["cnot", "cz", "swap", "cu"]:
                op = self._build_two_qubit_op(matrix, qubits[0], qubits[1])
                U = op @ U
        
        return U
    
    def _build_single_qubit_op(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Build full N-qubit operator for single-qubit gate."""
        ops = []
        for i in range(self.num_qubits - 1, -1, -1):
            if i == qubit:
                ops.append(gate)
            else:
                ops.append(PAULI_I)
        return kron(*ops)
    
    def _build_two_qubit_op(self, gate: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """Build full N-qubit operator for two-qubit gate."""
        left_ops = []
        right_ops = []
        
        for i in range(self.num_qubits):
            if i < min(q1, q2):
                left_ops.append(PAULI_I)
            elif i > max(q1, q2):
                right_ops.append(PAULI_I)
        
        left = kron(*left_ops) if left_ops else np.array([[1]], dtype=complex)
        right = kron(*right_ops) if right_ops else np.array([[1]], dtype=complex)
        
        if q1 < q2:
            return kron(left, gate, right)
        else:
            # Swap qubit order for the gate
            return kron(left, gate, right)
    
    def depth(self) -> int:
        """Calculate circuit depth (number of sequential layers)."""
        if not self.gates:
            return 0
        
        # Track which qubits are busy in each layer
        layers = []
        current_layer_qubits = set()
        
        for gate_type, qubits, _, _ in self.gates:
            qubits_set = set(qubits)
            
            # Check if any qubit is busy in current layer
            if qubits_set & current_layer_qubits:
                # Start new layer
                layers.append(current_layer_qubits)
                current_layer_qubits = qubits_set
            else:
                # Add to current layer
                current_layer_qubits |= qubits_set
        
        if current_layer_qubits:
            layers.append(current_layer_qubits)
        
        return len(layers)
    
    def gate_count(self) -> dict:
        """Count number of each gate type."""
        counts = {}
        for gate_type, _, _, name in self.gates:
            counts[gate_type] = counts.get(gate_type, 0) + 1
        return counts
    
    def __len__(self) -> int:
        """Return number of gates in circuit."""
        return len(self.gates)
    
    def __repr__(self) -> str:
        return f"QuantumCircuit(num_qubits={self.num_qubits}, gates={len(self.gates)}, depth={self.depth()})"
    
    def __str__(self) -> str:
        """Pretty print the circuit."""
        lines = [f"Quantum Circuit: {self.name}"]
        lines.append(f"Qubits: {self.num_qubits}")
        lines.append(f"Depth: {self.depth()}")
        lines.append(f"Gate Count: {self.gate_count()}")
        lines.append("\nGate Sequence:")
        for i, (gate_type, qubits, _, name) in enumerate(self.gates):
            q_str = ",".join(map(str, qubits))
            lines.append(f"  {i+1}. {name} on qubit(s) [{q_str}]")
        return "\n".join(lines)
