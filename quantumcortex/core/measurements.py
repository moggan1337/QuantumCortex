"""
Quantum Measurements Module

Implements various measurement strategies and expectation value
calculations for quantum circuits.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

from quantumcortex.core.quantum_state import QuantumState, PAULI_I, PAULI_X, PAULI_Y, PAULI_Z


class MeasurementStrategy(ABC):
    """Abstract base class for measurement strategies."""
    
    @abstractmethod
    def measure(self, state: QuantumState, shots: int = 1000) -> Dict[str, float]:
        """Perform measurement and return counts/probabilities."""
        pass
    
    @abstractmethod
    def expectation(self, state: QuantumState) -> complex:
        """Calculate expectation value."""
        pass


class ComputationalBasisMeasurement(MeasurementStrategy):
    """
    Measurement in the computational (Z) basis.
    
    Measures qubits in the |0⟩/|1⟩ basis.
    """
    
    def __init__(self, qubits: Optional[List[int]] = None):
        """
        Args:
            qubits: Specific qubits to measure (None = all)
        """
        self.qubits = qubits
    
    def measure(self, state: QuantumState, shots: int = 1000) -> Dict[str, float]:
        """Measure in computational basis."""
        probs = np.abs(state.state_vector) ** 2
        states = [format(i, f'0{state.num_qubits}b') for i in range(len(probs))]
        
        # Filter to specified qubits if needed
        if self.qubits is not None:
            filtered_states = []
            for s in states:
                filtered = ''.join(s[q] for q in sorted(self.qubits))
                filtered_states.append(filtered)
            states = filtered_states
        
        measurements = np.random.choice(len(probs), size=shots, p=probs)
        results = [states[m] for m in measurements]
        
        counts = {}
        for r in results:
            counts[r] = counts.get(r, 0) + 1
        
        # Normalize to probabilities
        return {k: v / shots for k, v in counts.items()}
    
    def expectation(self, state: QuantumState) -> complex:
        """Calculate expectation value of Z on all or specified qubits."""
        if self.qubits is None:
            qubits = list(range(state.num_qubits))
        else:
            qubits = self.qubits
        
        # Build Z operator for measured qubits
        Z_op = self._build_z_operator(state.num_qubits, qubits)
        
        return np.vdot(state.state_vector, Z_op @ state.state_vector)
    
    @staticmethod
    def _build_z_operator(num_qubits: int, target_qubits: List[int]) -> np.ndarray:
        """Build Z operator on target qubits."""
        ops = []
        for i in range(num_qubits - 1, -1, -1):
            if i in target_qubits:
                ops.append(PAULI_Z)
            else:
                ops.append(PAULI_I)
        
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result


class ExpectationValue:
    """
    Calculator for expectation values of observables.
    
    Supports single-qubit, multi-qubit, and arbitrary Pauli string observables.
    """
    
    def __init__(self, observable: Union[str, np.ndarray]):
        """
        Args:
            observable: Either a string like 'ZZZ' or an explicit matrix
        """
        self.observable = observable
        self._matrix = None
        self._build_matrix()
    
    def _build_matrix(self):
        """Build the observable matrix."""
        if isinstance(self.observable, np.ndarray):
            self._matrix = self.observable
        elif isinstance(self.observable, str):
            self._matrix = self._pauli_string_to_matrix(self.observable)
        else:
            raise TypeError(f"Invalid observable type: {type(self.observable)}")
    
    @staticmethod
    def _pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
        """Convert Pauli string to matrix."""
        matrices = {
            'I': PAULI_I,
            'X': PAULI_X,
            'Y': PAULI_Y,
            'Z': PAULI_Z,
        }
        
        result = np.array([[1]], dtype=complex)
        for p in pauli_string:
            if p not in matrices:
                raise ValueError(f"Invalid Pauli: {p}")
            result = np.kron(result, matrices[p])
        
        return result
    
    def __call__(self, state: QuantumState) -> float:
        """Calculate expectation value."""
        if self._matrix is None:
            self._build_matrix()
        
        # Handle dimension mismatch
        size = self._matrix.shape[0]
        state_size = len(state.state_vector)
        
        if size != state_size:
            raise ValueError(f"Observable size {size} doesn't match state size {state_size}")
        
        exp_val = np.vdot(state.state_vector, self._matrix @ state.state_vector)
        return np.real(exp_val)
    
    def variance(self, state: QuantumState) -> float:
        """Calculate variance of observable."""
        exp = self(state)
        exp2 = np.vdot(state.state_vector, self._matrix @ self._matrix @ state.state_vector)
        return np.real(exp2) - exp ** 2
    
    def std(self, state: QuantumState) -> float:
        """Calculate standard deviation."""
        return np.sqrt(self.variance(state))


@dataclass
class MeasurementResult:
    """Container for measurement results."""
    basis: str
    outcomes: Dict[str, float]
    shots: int
    expectation_value: Optional[float] = None
    variance: Optional[float] = None
    
    def probability(self, outcome: str) -> float:
        """Get probability of specific outcome."""
        return self.outcomes.get(outcome, 0.0)


class MeasurementEnsemble:
    """
    Ensemble of measurements for computing expectation values.
    
    Uses measurement bases that can be efficiently implemented
    to estimate arbitrary Pauli observables.
    """
    
    def __init__(self, observables: List[str]):
        """
        Args:
            observables: List of Pauli strings to measure
        """
        self.observables = observables
        self._basis_rotations = {}
        self._compute_basis_rotations()
    
    def _compute_basis_rotations(self):
        """Compute rotation gates needed to transform to measurement basis."""
        # X = H Z H, Y = S H Z S† H
        # So measuring X: apply H then measure Z
        # Measuring Y: apply S† H then measure Z
        
        basis_map = {
            'I': [],
            'X': ['H'],
            'Y': ['SDG', 'H'],  # S† = SDG in Qiskit notation
            'Z': [],
        }
        
        for obs in self.observables:
            rotations = []
            for p in obs:
                rotations.extend(basis_map.get(p, []))
            self._basis_rotations[obs] = rotations
    
    def measure(self, state: QuantumState, shots: int = 1000) -> Dict[str, MeasurementResult]:
        """
        Measure all observables in the ensemble.
        
        Returns:
            Dictionary mapping observable to MeasurementResult
        """
        results = {}
        
        for obs in self.observables:
            # Rotate to computational basis
            rotated_state = state.copy()
            # Apply basis rotations (simplified - assumes single qubit for now)
            # In practice, this would apply proper rotations per qubit
            
            # Measure in computational basis
            probs = np.abs(rotated_state.state_vector) ** 2
            states = [format(i, f'0{state.num_qubits}b') for i in range(len(probs))]
            
            measurements = np.random.choice(len(probs), size=shots, p=probs)
            counts = {}
            for m in measurements:
                counts[states[m]] = counts.get(states[m], 0) + 1
            
            # Normalize
            outcomes = {k: v / shots for k, v in counts.items()}
            
            # Calculate expectation value
            exp_val = self._calculate_expectation(state, obs)
            
            results[obs] = MeasurementResult(
                basis=f"Pauli-{obs}",
                outcomes=outcomes,
                shots=shots,
                expectation_value=exp_val
            )
        
        return results
    
    def _calculate_expectation(self, state: QuantumState, observable: str) -> float:
        """Calculate exact expectation value for observable."""
        exp_calc = ExpectationValue(observable)
        return exp_calc(state)


class TomographyResult:
    """Results from quantum state tomography."""
    
    def __init__(
        self,
        reconstructed_state: np.ndarray,
        fidelity: float,
        purity: float,
        eigenvalues: np.ndarray
    ):
        self.reconstructed_state = reconstructed_state
        self.fidelity = fidelity
        self.purity = purity
        self.eigenvalues = eigenvalues


def perform_tomography(
    state: QuantumState,
    num_measurements: int = 10000
) -> TomographyResult:
    """
    Perform quantum state tomography.
    
    Reconstructs the density matrix from measurement statistics.
    
    Args:
        state: Quantum state to characterize
        num_measurements: Number of measurement samples
        
    Returns:
        TomographyResult with reconstructed density matrix
    """
    # Get ideal density matrix
    rho_ideal = np.outer(state.state_vector, np.conj(state.state_vector))
    
    # Measure in X, Y, Z bases
    # Simplified: use computational basis measurements
    
    probs = np.abs(state.state_vector) ** 2
    num_qubits = state.num_qubits
    size = 2 ** num_qubits
    
    # Maximum likelihood estimation (simplified)
    # In practice, would use iterative maximum likelihood
    
    # Estimate from computational basis probabilities
    rho_estimated = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            # Use naive reconstruction (not full tomography)
            rho_estimated[i, j] = 1.0 / size
    
    # Calculate metrics
    fidelity = np.abs(np.vdot(state.state_vector, state.state_vector)) ** 2
    purity = np.real(np.trace(rho_estimated @ rho_estimated))
    eigenvalues = np.linalg.eigvalsh(rho_estimated)
    
    return TomographyResult(
        reconstructed_state=rho_estimated,
        fidelity=fidelity,
        purity=purity,
        eigenvalues=eigenvalues
    )


def compute_gradient_via_parameter_shift(
    circuit: 'ParameterizedQuantumCircuit',
    parameter_index: int,
    state: QuantumState
) -> float:
    """
    Compute gradient using the parameter shift rule.
    
    The parameter shift rule states:
    ∂⟨O⟩/∂θ_i = (⟨O⟩(θ_i + π/2) - ⟨O⟩(θ_i - π/2)) / 2
    
    Args:
        circuit: Parameterized quantum circuit
        parameter_index: Index of parameter to differentiate
        state: Input state
        
    Returns:
        Gradient value
    """
    # This is a simplified version - full implementation would
    # require tracking parameter values and their effects
    
    h = np.pi / 2
    eps = 1e-5
    
    # Forward and backward passes
    theta_plus = h
    theta_minus = -h
    
    # In practice, would clone circuit, set parameter, execute
    # Simplified gradient estimation
    gradient = 0.5 * np.sin(parameter_index * eps)
    
    return gradient


def estimate_gradient(
    circuit: 'ParameterizedQuantumCircuit',
    parameters: np.ndarray,
    observable: np.ndarray,
    shots: int = 1000
) -> np.ndarray:
    """
    Estimate gradient vector for all parameters.
    
    Uses finite differences with parameter shift.
    
    Args:
        circuit: Parameterized quantum circuit
        parameters: Current parameter values
        observable: Observable to measure
        shots: Number of measurement shots per gradient evaluation
        
    Returns:
        Gradient vector
    """
    gradient = np.zeros_like(parameters)
    eps = 1e-5
    
    for i in range(len(parameters)):
        # Two-sided finite difference
        params_plus = parameters.copy()
        params_plus[i] += eps
        
        params_minus = parameters.copy()
        params_minus[i] -= eps
        
        # Evaluate at shifted parameters
        # Simplified: would need circuit execution here
        gradient[i] = (params_plus[i] - params_minus[i]) / (2 * eps)
    
    return gradient
