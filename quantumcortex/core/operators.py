"""
Quantum Operators Module

Implements various quantum gates, operators, and Pauli strings
for the QuantumCortex framework.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from abc import ABC, abstractmethod


class QuantumOperator(ABC):
    """Abstract base class for quantum operators."""
    
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the matrix representation."""
        pass
    
    @abstractmethod
    def num_qubits(self) -> int:
        """Return number of qubits this operator acts on."""
        pass
    
    def __call__(self, state: 'QuantumState') -> 'QuantumState':
        """Apply operator to a quantum state."""
        from quantumcortex.core.quantum_state import QuantumState
        matrix = self.matrix()
        if self.num_qubits() == 1:
            return state.apply_gate(matrix, [0])
        else:
            # For multi-qubit operators, would need target specification
            return state.apply_gate(matrix, list(range(self.num_qubits())))


class Gate(QuantumOperator):
    """
    Represents a quantum gate.
    
    Attributes:
        name: Name of the gate
        matrix: Gate matrix
        num_qubits: Number of qubits the gate acts on
        parameters: Optional parameters for parameterized gates
    """
    
    def __init__(
        self,
        name: str,
        matrix: np.ndarray,
        num_qubits: int = 1,
        parameters: Optional[Tuple] = None
    ):
        self.name = name
        self._matrix = matrix
        self._num_qubits = num_qubits
        self.parameters = parameters
    
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    def num_qubits(self) -> int:
        return self._num_qubits
    
    def __repr__(self) -> str:
        return f"Gate({self.name}, qubits={self._num_qubits})"


class PauliGate(Gate):
    """Pauli X, Y, Z gates."""
    
    def __init__(self, pauli: str):
        if pauli not in ['X', 'Y', 'Z', 'I']:
            raise ValueError(f"Invalid Pauli: {pauli}")
        
        matrices = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }
        
        super().__init__(f"Pauli{pauli}", matrices[pauli], num_qubits=1)
        self.pauli = pauli


class RotationGate(Gate):
    """Parameterized rotation gates Rx, Ry, Rz."""
    
    def __init__(self, axis: str, theta: float):
        if axis not in ['x', 'y', 'z']:
            raise ValueError(f"Invalid axis: {axis}")
        
        self.axis = axis
        self.theta = theta
        
        cos_t2 = np.cos(theta / 2)
        sin_t2 = np.sin(theta / 2)
        
        if axis == 'x':
            matrix = np.array([
                [cos_t2, -1j * sin_t2],
                [-1j * sin_t2, cos_t2]
            ], dtype=complex)
        elif axis == 'y':
            matrix = np.array([
                [cos_t2, -sin_t2],
                [sin_t2, cos_t2]
            ], dtype=complex)
        else:  # z
            matrix = np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ], dtype=complex)
        
        super().__init__(f"R{axis}({theta:.4f})", matrix, num_qubits=1, parameters=(theta,))


class HadamardGate(Gate):
    """Hadamard gate for creating superpositions."""
    
    def __init__(self):
        matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        super().__init__("H", matrix, num_qubits=1)


class PauliString:
    """
    Represents a Pauli string operator (tensor product of Pauli matrices).
    
    Example: "XYZI" represents X⊗Y⊗Z⊗I
    """
    
    def __init__(self, paulis: Union[str, List[str]]):
        if isinstance(paulis, str):
            self.paulis = list(paulis)
        else:
            self.paulis = list(paulis)
        
        self._matrix = None
        self._build_matrix()
    
    def _build_matrix(self):
        """Build the matrix representation of the Pauli string."""
        matrices = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }
        
        result = np.array([[1]], dtype=complex)
        for p in self.paulis:
            result = np.kron(result, matrices[p])
        
        self._matrix = result
    
    def matrix(self) -> np.ndarray:
        return self._matrix
    
    def num_qubits(self) -> int:
        return len(self.paulis)
    
    def __mul__(self, other: 'PauliString') -> 'PauliString':
        """Multiply two Pauli strings."""
        new_paulis = self.paulis + other.paulis
        return PauliString(new_paulis)
    
    def __pow__(self, n: int) -> 'PauliString':
        """Raise Pauli string to power n."""
        return PauliString(self.paulis * n)
    
    def __str__(self) -> str:
        return "".join(self.paulis)
    
    def __repr__(self) -> str:
        return f"PauliString('{''.join(self.paulis)}')"


class Hamiltonian:
    """
    Represents a Hamiltonian as a sum of Pauli strings.
    
    H = Σ_i c_i P_i
    
    Attributes:
        terms: Dictionary mapping PauliString to coefficient
    """
    
    def __init__(self, terms: Optional[dict] = None):
        self.terms = terms or {}
        self._matrix = None
    
    def add_term(self, pauli_string: PauliString, coefficient: complex = 1.0):
        """Add a term to the Hamiltonian."""
        key = str(pauli_string)
        if key in self.terms:
            self.terms[key] += coefficient
        else:
            self.terms[key] = coefficient
        self._matrix = None
    
    def matrix(self) -> np.ndarray:
        """Compute the full Hamiltonian matrix."""
        if self._matrix is not None:
            return self._matrix
        
        if not self.terms:
            return np.zeros((1, 1), dtype=complex)
        
        # Get number of qubits from first term
        first_key = list(self.terms.keys())[0]
        num_qubits = len(first_key)
        size = 2 ** num_qubits
        
        H = np.zeros((size, size), dtype=complex)
        
        for pauli_str, coeff in self.terms.items():
            ps = PauliString(pauli_str)
            H += coeff * ps.matrix()
        
        self._matrix = H
        return H
    
    def eigenvalues(self) -> np.ndarray:
        """Get eigenvalues of the Hamiltonian."""
        return np.linalg.eigvalsh(self.matrix())
    
    def ground_state_energy(self) -> float:
        """Get the ground state energy."""
        return np.min(self.eigenvalues())
    
    def expectation(self, state: 'QuantumState') -> complex:
        """Calculate expectation value of Hamiltonian."""
        matrix = self.matrix()
        return np.vdot(state.state_vector, matrix @ state.state_vector)
    
    @classmethod
    def from_observable(cls, observable: np.ndarray) -> 'Hamiltonian':
        """Create Hamiltonian from explicit matrix."""
        h = cls()
        h._matrix = observable
        return h
    
    @classmethod
    def transverse_ising(
        cls,
        num_sites: int,
        coupling: float = 1.0,
        transverse_field: float = 1.0
    ) -> 'Hamiltonian':
        """
        Create transverse-field Ising model Hamiltonian.
        
        H = -J Σ Z_i Z_{i+1} - h Σ X_i
        """
        h = cls()
        
        # ZZ interactions
        for i in range(num_sites - 1):
            pauli_str = ['I'] * num_sites
            pauli_str[i] = 'Z'
            pauli_str[i + 1] = 'Z'
            h.add_term(PauliString(pauli_str), -coupling)
        
        # Transverse field (X terms)
        for i in range(num_sites):
            pauli_str = ['I'] * num_sites
            pauli_str[i] = 'X'
            h.add_term(PauliString(pauli_str), -transverse_field)
        
        return h
    
    @classmethod
    def heisenberg(
        cls,
        num_sites: int,
        J: float = 1.0,
        h: float = 0.0
    ) -> 'Hamiltonian':
        """
        Create Heisenberg model Hamiltonian.
        
        H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
        """
        hamiltonian = cls()
        
        for i in range(num_sites - 1):
            for pauli in ['X', 'Y', 'Z']:
                pauli_str = ['I'] * num_sites
                pauli_str[i] = pauli
                pauli_str[i + 1] = pauli
                hamiltonian.add_term(PauliString(pauli_str), J)
        
        # Optional magnetic field
        if h != 0:
            for i in range(num_sites):
                pauli_str = ['I'] * num_sites
                pauli_str[i] = 'Z'
                hamiltonian.add_term(PauliString(pauli_str), h)
        
        return hamiltonian
    
    def __add__(self, other: 'Hamiltonian') -> 'Hamiltonian':
        """Add two Hamiltonians."""
        new_terms = self.terms.copy()
        for key, coeff in other.terms.items():
            if key in new_terms:
                new_terms[key] += coeff
            else:
                new_terms[key] = coeff
        return Hamiltonian(new_terms)
    
    def __mul__(self, scalar: complex) -> 'Hamiltonian':
        """Multiply Hamiltonian by scalar."""
        new_terms = {k: v * scalar for k, v in self.terms.items()}
        return Hamiltonian(new_terms)
    
    def __rmul__(self, scalar: complex) -> 'Hamiltonian':
        return self.__mul__(scalar)
    
    def __str__(self) -> str:
        terms = [f"{coeff:.3f}*{key}" for key, coeff in self.terms.items()]
        return " + ".join(terms) if terms else "0"
    
    def __repr__(self) -> str:
        return f"Hamiltonian(terms={len(self.terms)})"


class MeasurementBasis:
    """Measurement basis transformation."""
    
    @staticmethod
    def computational() -> Tuple[np.ndarray, List[str]]:
        """Z-basis (computational basis)."""
        basis_vectors = [
            np.array([1, 0], dtype=complex),  # |0⟩
            np.array([0, 1], dtype=complex),  # |1⟩
        ]
        return np.array(basis_vectors), ['0', '1']
    
    @staticmethod
    def hadamard() -> Tuple[np.ndarray, List[str]]:
        """Hadamard basis (X-basis)."""
        h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        basis_vectors = [
            h @ np.array([1, 0], dtype=complex),  # |+⟩
            h @ np.array([0, 1], dtype=complex),  # |-⟩
        ]
        return np.array(basis_vectors), ['+', '-']
    
    @staticmethod
    def y_basis() -> Tuple[np.ndarray, List[str]]:
        """Y-basis."""
        basis_vectors = [
            np.array([1, 1j], dtype=complex) / np.sqrt(2),   # |+i⟩
            np.array([1, -1j], dtype=complex) / np.sqrt(2),  # |-i⟩
        ]
        return np.array(basis_vectors), ['+i', '-i']


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute {A, B} = AB + BA."""
    return A @ B + B @ A


def is_hermitian(M: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is Hermitian."""
    return np.allclose(M, np.conj(M.T), atol=tol)


def is_unitary(M: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if matrix is unitary."""
    return np.allclose(M @ np.conj(M.T), np.eye(len(M)), atol=tol)


def spectral_decomposition(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral decomposition of Hermitian matrix.
    
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvalues, eigenvectors


def matrix_exponential(M: np.ndarray) -> np.ndarray:
    """Compute matrix exponential e^M."""
    return np.linalg.matrix_exp(M)


def matrix_logarithm(U: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm of unitary matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(U)
    log_eigenvalues = np.log(eigenvalues + 1e-15)
    return eigenvectors @ np.diag(log_eigenvalues) @ np.conj(eigenvectors.T)
