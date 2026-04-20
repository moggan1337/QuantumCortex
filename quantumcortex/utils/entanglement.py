"""
Entanglement Analysis Module

Tools for analyzing and measuring quantum entanglement
in neural network states.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass
import scipy.linalg as la

from quantumcortex.core.quantum_state import QuantumState, PAULI_I, PAULI_X, PAULI_Y, PAULI_Z


@dataclass
class EntanglementMetric:
    """Container for entanglement metrics."""
    name: str
    value: float
    details: Optional[Dict] = None


class EntanglementAnalyzer:
    """
    Analyzer for quantum entanglement properties.
    
    Provides various measures of entanglement including
    entropy, concurrence, and negativity.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
    
    def compute_all_metrics(
        self,
        state: QuantumState
    ) -> List[EntanglementMetric]:
        """
        Compute all available entanglement metrics.
        
        Args:
            state: Quantum state to analyze
            
        Returns:
            List of EntanglementMetric objects
        """
        metrics = []
        
        # Bipartite entanglement for different partitions
        for i in range(self.num_qubits - 1):
            partition = list(range(i + 1))
            entropy = compute_entanglement_entropy(state, partition)
            metrics.append(EntanglementMetric(
                name=f"Entropy_Subsystem_{i}",
                value=entropy,
                details={'partition': partition}
            ))
        
        # Concurrence (for 2-qubit states)
        if self.num_qubits == 2:
            concurrence = self._compute_concurrence(state)
            metrics.append(EntanglementMetric(
                name="Concurrence",
                value=concurrence
            ))
        
        # Negativity
        negativity = self._compute_negativity(state)
        metrics.append(EntanglementMetric(
            name="Negativity",
            value=negativity
        ))
        
        # Entanglement of formation (for 2 qubits)
        if self.num_qubits == 2:
            eof = self._entanglement_of_formation(state)
            metrics.append(EntanglementMetric(
                name="Entanglement_of_Formation",
                value=eof
            ))
        
        return metrics
    
    def compute_entropy(
        self,
        state: QuantumState,
        partition: List[int]
    ) -> float:
        """Compute von Neumann entropy of reduced density matrix."""
        return compute_entanglement_entropy(state, partition)
    
    def _compute_concurrence(self, state: QuantumState) -> float:
        """
        Compute concurrence for 2-qubit state.
        
        Concurrence = max(0, λ₁ - λ₂ - λ₃ - λ₄)
        
        where λᵢ are square roots of eigenvalues of R = ρρ̃
        in descending order.
        """
        if state.num_qubits != 2:
            return 0.0
        
        # Get density matrix
        rho = np.outer(state.state_vector, np.conj(state.state_vector))
        
        # Spin-flipped state
        sigma_y = PAULI_Y
        R = rho @ (sigma_y @ np.conj(rho) @ sigma_y)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sqrt(np.abs(eigenvalues))
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Concurrence
        concurrence = max(0, eigenvalues[0] - sum(eigenvalues[1:]))
        
        return concurrence
    
    def _compute_negativity(self, state: QuantumState) -> float:
        """
        Compute negativity using PPT criterion.
        
        Negativity = (||ρ^{T_A}||₁ - 1) / 2
        """
        if state.num_qubits < 2:
            return 0.0
        
        # Get density matrix
        rho = np.outer(state.state_vector, np.conj(state.state_vector))
        
        # Partial transpose (transpose first qubit)
        rho_pt = self._partial_transpose(rho, [0], state.num_qubits)
        
        # Sum of absolute values of negative eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        negativity = np.sum(np.abs(eigenvalues) - eigenvalues) / 2
        
        return negativity
    
    def _partial_transpose(
        self,
        rho: np.ndarray,
        transpose_qubits: List[int],
        num_qubits: int
    ) -> np.ndarray:
        """Compute partial transpose of density matrix."""
        # Reshape to tensor form
        shape = [2] * (2 * num_qubits)
        rho_tensor = rho.reshape(shape)
        
        # Transpose specified qubits
        perm = list(range(2 * num_qubits))
        for q in transpose_qubits:
            perm[q], perm[q + num_qubits] = perm[q + num_qubits], perm[q]
        
        rho_pt = np.transpose(rho_tensor, perm)
        
        return rho_pt.reshape(rho.shape)
    
    def _entanglement_of_formation(self, state: QuantumState) -> float:
        """
        Compute entanglement of formation for 2-qubit state.
        
        E = -c*log₂(c) - (1-c)*log₂(1-c)
        
        where c is the concurrence.
        """
        c = self._compute_concurrence(state)
        
        if c == 0:
            return 0.0
        
        # Formula for entanglement of formation
        h = -(c * np.log2(c) + (1 - c) * np.log2(1 - c))
        
        return h
    
    def schmidt_decomposition(
        self,
        state: QuantumState,
        partition_a: List[int],
        partition_b: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Schmidt decomposition.
        
        Args:
            state: Bipartite state
            partition_a: Qubits in partition A
            partition_b: Qubits in partition B (auto-computed if None)
            
        Returns:
            Tuple of (Schmidt coefficients, Schmidt vectors)
        """
        if partition_b is None:
            partition_b = [i for i in range(state.num_qubits) if i not in partition_a]
        
        # Get reduced density matrix
        rho_a = state.partial_trace(partition_b)
        
        # Eigenvalues are squared Schmidt coefficients
        eigenvalues, eigenvectors = np.linalg.eigh(rho_a)
        eigenvalues = np.abs(eigenvalues)
        
        # Schmidt coefficients
        schmidt_coeffs = np.sqrt(eigenvalues[eigenvalues > 1e-10])
        
        return schmidt_coeffs, eigenvectors
    
    def is_entangled(self, state: QuantumState) -> bool:
        """Check if state is entangled using PPT criterion."""
        if state.num_qubits < 2:
            return False
        
        negativity = self._compute_negativity(state)
        return negativity > 1e-6
    
    def purity_bipartite(self, state: QuantumState, partition: List[int]) -> float:
        """
        Compute purity of reduced state.
        
        P = Tr(ρ_A²)
        
        P = 1 for pure states
        P = 1/d for maximally mixed states
        """
        rho_a = state.partial_trace(partition)
        return np.real(np.trace(rho_a @ rho_a))


def compute_entanglement_entropy(
    state: Union[QuantumState, np.ndarray],
    partition: List[int]
) -> float:
    """
    Compute entanglement entropy using Schmidt decomposition.
    
    Args:
        state: Quantum state or state vector
        partition: Qubits to trace out
        
    Returns:
        Von Neumann entropy of reduced state
    """
    if isinstance(state, np.ndarray):
        if state.ndim == 1:
            # State vector
            rho = np.outer(state, np.conj(state))
        else:
            # Density matrix
            rho = state
    else:
        rho = state.partial_trace(partition)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Von Neumann entropy
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
    
    return entropy


def measure_bipartite_entanglement(
    state: QuantumState,
    qubit_a: int,
    qubit_b: int
) -> Dict[str, float]:
    """
    Measure entanglement between two specific qubits.
    
    Returns multiple entanglement measures.
    """
    # Get reduced density matrix for the two qubits
    other_qubits = [i for i in range(state.num_qubits) if i != qubit_a and i != qubit_b]
    rho_ab = state.partial_trace(other_qubits)
    
    # Compute measures
    # Entanglement entropy
    eigenvalues = np.linalg.eigvalsh(rho_ab)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
    
    # Purity
    purity = np.real(np.trace(rho_ab @ rho_ab))
    
    # Negativity
    # (partial transpose would need more complex implementation)
    
    return {
        'entropy': entropy,
        'purity': purity,
        'max_entropy': np.log2(4),  # Max for 2 qubits
        'relative_entropy': entropy / np.log2(4)  # Normalized
    }


def concurrence(state: np.ndarray) -> float:
    """
    Compute concurrence for a 2-qubit state.
    
    Args:
        state: 2-qubit state vector or density matrix
        
    Returns:
        Concurrence value in [0, 1]
    """
    if state.ndim == 1:
        rho = np.outer(state, np.conj(state))
    else:
        rho = state
    
    # Spin-flipped state
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_y_kron = np.kron(sigma_y, sigma_y)
    
    # R = ρ σ_y ρ* σ_y
    R = rho @ sigma_y_kron @ np.conj(rho) @ sigma_y_kron
    
    # Square roots of eigenvalues
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sqrt(np.abs(eigvals))
    eigvals = np.sort(eigvals)[::-1]
    
    # Concurrence
    return max(0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3])


def negativity(state: np.ndarray, subsystem: int = 0) -> float:
    """
    Compute negativity using partial transpose.
    
    Args:
        state: Density matrix
        subsystem: Subsystem to trace over (0-indexed)
        
    Returns:
        Negativity value
    """
    num_qubits = int(np.log2(state.shape[0]))
    
    # Reshape to tensor
    shape = [2] * (2 * num_qubits)
    rho_tensor = state.reshape(shape)
    
    # Partial transpose over specified subsystem
    perm = list(range(2 * num_qubits))
    perm[subsystem], perm[subsystem + num_qubits] = \
        perm[subsystem + num_qubits], perm[subsystem]
    
    rho_pt = np.transpose(rho_tensor, perm).reshape(state.shape)
    
    # Sum of absolute values of negative eigenvalues
    eigvals = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigvals) - eigvals) / 2


def mutual_information(
    state: QuantumState,
    subsystem_a: List[int],
    subsystem_b: List[int]
) -> float:
    """
    Compute quantum mutual information I(A:B).
    
    I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
    """
    # Full system entropy
    S_ab = compute_entanglement_entropy(state, [])
    
    # Subsystem entropies
    S_a = compute_entanglement_entropy(state, subsystem_a)
    S_b = compute_entanglement_entropy(state, subsystem_b)
    
    return S_a + S_b - S_ab


def discord(state: QuantumState, subsystem_a: List[int]) -> float:
    """
    Compute quantum discord.
    
    Discord = S(ρ_A) - S(ρ) + min_C { S(ρ|C) }
    
    where C is a measurement on subsystem B.
    
    Note: This is a simplified approximation.
    """
    # For a simplified version, compute geometric discord
    rho = np.outer(state.state_vector, np.conj(state.state_vector))
    
    # Reduced density matrix
    rho_a = state.partial_trace([i for i in range(state.num_qubits) if i not in subsystem_a])
    
    # Classical correlation (simplified)
    # In practice, would optimize over measurement bases
    classical_corr = 0.0
    
    # Quantum discord approximation
    S_a = compute_entanglement_entropy(state, subsystem_a)
    S_full = state.von_neumann_entropy()
    
    discord_val = S_a - S_full + classical_corr
    
    return max(0, discord_val)


class EntanglementWitness:
    """
    Entanglement witness operators.
    
    Used to detect entanglement without full state tomography.
    """
    
    @staticmethod
    def pauli_witness(qubits: List[int], paulis: str) -> np.ndarray:
        """
        Create Pauli-based entanglement witness.
        
        W = P for some Pauli string P
        Tr(W ρ) < 0 indicates entanglement
        """
        matrices = {
            'I': PAULI_I,
            'X': PAULI_X,
            'Y': PAULI_Y,
            'Z': PAULI_Z,
        }
        
        witness = np.array([[1]], dtype=complex)
        for p in paulis:
            witness = np.kron(witness, matrices.get(p, PAULI_I))
        
        return witness
    
    @staticmethod
    def test_witness(
        state: QuantumState,
        witness: np.ndarray
    ) -> float:
        """Test entanglement witness on a state."""
        rho = np.outer(state.state_vector, np.conj(state.state_vector))
        return np.real(np.trace(witness @ rho))
    
    @staticmethod
    def is_witness_operator(matrix: np.ndarray) -> bool:
        """Check if matrix is a valid entanglement witness."""
        # W is witness if Tr(W) = 1 and W is Hermitian
        return (
            np.isclose(np.trace(matrix), 1.0) and
            np.allclose(matrix, np.conj(matrix.T))
        )
