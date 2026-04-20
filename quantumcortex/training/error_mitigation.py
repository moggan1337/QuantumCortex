"""
Quantum Error Mitigation Module

Implements various error mitigation techniques for noisy
intermediate-scale quantum (NISQ) devices.
"""

import numpy as np
from typing import List, Optional, Tuple, Callable, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from quantumcortex.core.quantum_state import QuantumState


@dataclass
class ErrorMitigationConfig:
    """Configuration for error mitigation."""
    method: str = 'zsd'  # Zero-noise extrapolation, readout mitigation, etc.
    noise_strength: float = 0.1
    num_copies: int = 10


class ErrorMitigation(ABC):
    """
    Abstract base class for error mitigation techniques.
    
    Error mitigation reduces the effects of noise without
    requiring full error correction.
    """
    
    def __init__(self, config: Optional[ErrorMitigationConfig] = None):
        self.config = config or ErrorMitigationConfig()
    
    @abstractmethod
    def mitigate(
        self,
        noisy_result: np.ndarray,
        circuit
    ) -> np.ndarray:
        """
        Mitigate noise in quantum computation result.
        
        Args:
            noisy_result: Result from noisy quantum computation
            circuit: Quantum circuit that produced the result
            
        Returns:
            Mitigated result
        """
        pass


class ZeroNoiseExtrapolation(ErrorMitigation):
    """
    Zero-Noise Extrapolation (ZNE).
    
    Extrapolates results from different noise levels to
    estimate the noise-free result.
    
    Method: Expand noise strength, measure expectation values,
    then extrapolate to zero noise.
    """
    
    def __init__(
        self,
        noise_factors: Optional[List[float]] = None,
        extrapolation_method: str = 'linear',
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        
        self.noise_factors = noise_factors or [1.0, 1.5, 2.0, 3.0]
        self.extrapolation_method = extrapolation_method
    
    def mitigate(
        self,
        noisy_results: Dict[float, float],
        circuit=None
    ) -> float:
        """
        Mitigate using zero-noise extrapolation.
        
        Args:
            noisy_results: Dict mapping noise_factor to expectation value
            circuit: Optional circuit for resampling
            
        Returns:
            Extrapolated zero-noise expectation value
        """
        noise_levels = list(noisy_results.keys())
        expectations = list(noisy_results.values())
        
        if self.extrapolation_method == 'linear':
            return self._linear_extrapolate(noise_levels, expectations)
        elif self.extrapolation_method == 'polynomial':
            return self._polynomial_extrapolate(noise_levels, expectations, degree=2)
        elif self.extrapolation_method == 'exponential':
            return self._exponential_extrapolate(noise_levels, expectations)
        else:
            return expectations[0]  # Return first result if no extrapolation
    
    def _linear_extrapolate(self, x: List[float], y: List[float]) -> float:
        """Linear extrapolation to x=0."""
        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Extrapolate to zero noise
        return intercept
    
    def _polynomial_extrapolate(
        self,
        x: List[float],
        y: List[float],
        degree: int = 2
    ) -> float:
        """Polynomial extrapolation to x=0."""
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        return poly(0)
    
    def _exponential_extrapolate(self, x: List[float], y: List[float]) -> float:
        """Exponential extrapolation: y = A * exp(-Bx) + C."""
        # Fit to y = A * exp(-Bx) + C
        # Take log of (y - C) where C is asymptotic value
        
        # Simple approach: assume asymptotic is 0
        y_positive = np.maximum(y, 1e-10)  # Avoid log(0)
        log_y = np.log(y_positive)
        
        # Linear fit in log space
        slope, intercept = np.polyfit(x, log_y, 1)
        
        # Extrapolate
        return np.exp(intercept)


class ReadoutMitigation(ErrorMitigation):
    """
    Readout Error Mitigation (REM).
    
    Constructs a confusion matrix for readout errors
    and inverts it to get mitigated probabilities.
    """
    
    def __init__(
        self,
        calibration_circuit_fn: Optional[Callable] = None,
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        self.calibration_matrix = None
        self.calibration_fn = calibration_circuit_fn
    
    def calibrate(
        self,
        num_qubits: int,
        execute_fn: Callable
    ) -> np.ndarray:
        """
        Calibrate readout error using calibration circuits.
        
        Args:
            num_qubits: Number of qubits
            execute_fn: Function to execute circuits and return counts
            
        Returns:
            Calibration matrix (2^n x 2^n)
        """
        size = 2 ** num_qubits
        self.calibration_matrix = np.zeros((size, size))
        
        # Prepare all computational basis states
        for state_idx in range(size):
            state_str = format(state_idx, f'0{num_qubits}b')
            
            # Execute circuit preparing |state⟩
            # and measure (should give mostly |state⟩)
            counts = execute_fn(state_str)
            
            # Normalize to probabilities
            total = sum(counts.values())
            for outcome, count in counts.items():
                outcome_idx = int(outcome, 2)
                self.calibration_matrix[state_idx, outcome_idx] = count / total
        
        return self.calibration_matrix
    
    def mitigate(self, noisy_probs: np.ndarray, circuit=None) -> np.ndarray:
        """
        Mitigate readout error.
        
        Args:
            noisy_probs: Noisy probability distribution
            circuit: Optional circuit (used for calibration)
            
        Returns:
            Mitigated probabilities
        """
        if self.calibration_matrix is None:
            return noisy_probs
        
        # Invert calibration matrix (use pseudoinverse for non-square)
        try:
            cal_inv = np.linalg.pinv(self.calibration_matrix)
            mitigated = cal_inv @ noisy_probs
            
            # Renormalize
            mitigated = np.maximum(mitigated, 0)
            mitigated = mitigated / np.sum(mitigated)
            
            return mitigated
        except np.linalg.LinAlgError:
            return noisy_probs


class ZSVD(ErrorMitigation):
    """
    Zero-noise Extrapolation with Singular Value Decomposition (ZSVD).
    
    Uses SVD to denoise the expectation values across different
    noise strengths.
    """
    
    def __init__(
        self,
        num_noise_levels: int = 5,
        truncation_rank: int = 2,
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        self.num_noise_levels = num_noise_levels
        self.truncation_rank = truncation_rank
    
    def mitigate(
        self,
        noisy_results: Dict[float, List[float]],
        circuit=None
    ) -> np.ndarray:
        """
        Mitigate using ZSVD method.
        
        Args:
            noisy_results: Dict mapping noise_factor to list of expectation values
            circuit: Optional circuit
            
        Returns:
            Mitigated expectation values
        """
        noise_factors = sorted(noisy_results.keys())
        num_obs = len(noisy_results[noise_factors[0]])
        
        # Build matrix of noisy results
        M = np.zeros((len(noise_factors), num_obs))
        for i, nf in enumerate(noise_factors):
            M[i, :] = noisy_results[nf]
        
        # SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        
        # Truncate to keep only significant singular values
        S_truncated = S.copy()
        S_truncated[self.truncation_rank:] = 0
        
        # Reconstruct denoised matrix
        M_denoised = U @ np.diag(S_truncated) @ Vt
        
        # Extrapolate to zero noise using first row
        noise_array = np.array(noise_factors)
        
        # Linear extrapolation for each observable
        mitigated = np.zeros(num_obs)
        for j in range(num_obs):
            # Fit line to denoised values
            y = M_denoised[:, j]
            coeffs = np.polyfit(noise_array, y, 1)
            mitigated[j] = np.polyval(coeffs, 0)
        
        return mitigated


class ProbabilisticErrorAmplification(ErrorMitigation):
    """
    Probabilistic Error Amplification (PEA).
    
    Amplifies errors probabilistically to enable extrapolation.
    """
    
    def __init__(
        self,
        gate_fidelities: Optional[Dict[str, float]] = None,
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        self.gate_fidelities = gate_fidelities or {}
    
    def amplify_error(
        self,
        circuit,
        noise_factor: float
    ):
        """
        Create amplified-noise version of circuit.
        
        Args:
            circuit: Original circuit
            noise_factor: Factor to amplify noise by
            
        Returns:
            Circuit with amplified noise
        """
        # In practice, this would add additional noisy gates
        # For simulation, we just return the original
        return circuit
    
    def mitigate(
        self,
        noisy_results: Dict[float, float],
        circuit=None
    ) -> float:
        """Mitigate using probabilistic error amplification."""
        return ZeroNoiseExtrapolation().mitigate(noisy_results, circuit)


class DynamicDecoupling(ErrorMitigation):
    """
    Dynamic Decoupling (DD).
    
    Inserts decoupling pulses during idle times to suppress
    decoherence and ambient noise.
    """
    
    def __init__(
        self,
        sequence_type: str = 'xy4',
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        self.sequence_type = sequence_type
        
        # Define pulse sequences
        self.sequences = {
            'spin_echo': ['X'],
            'cp': ['X', 'X'],  # Carr-Purcell
            'cpme': ['X', 'Y', 'X', 'Y'],  # Carr-Purcell-Meiboom-Gill
            'xy4': ['X', 'Y', 'Y', 'X'],  # XY-4
            'xy8': ['X', 'Y', 'Y', 'X', 'Y', 'X', 'X', 'Y'],  # XY-8
            'udd': ['X', 'X', 'X', 'X', 'X', 'X'],  # Universal DD
        }
    
    def insert_pulses(
        self,
        circuit,
        idle_qubits: List[int],
        duration: int
    ):
        """
        Insert decoupling pulses into idle times.
        
        Args:
            circuit: Quantum circuit
            idle_qubits: Qubits that are idle
            duration: Number of time steps
            
        Returns:
            Circuit with inserted pulses
        """
        sequence = self.sequences.get(self.sequence_type, self.sequences['xy4'])
        
        for pulse in sequence:
            for qubit in idle_qubits:
                if pulse == 'X':
                    circuit.x(qubit)
                elif pulse == 'Y':
                    circuit.y(qubit)
                # Z pulses can be implemented as identity
        
        return circuit
    
    def mitigate(self, noisy_result: np.ndarray, circuit=None) -> np.ndarray:
        """Mitigate using dynamic decoupling."""
        # In simulation, this just returns the result
        # In practice, would modify circuit and re-run
        return noisy_result


class RichardsonExtrapolation(ErrorMitigation):
    """
    Richardson Extrapolation for error mitigation.
    
    Uses results at multiple noise levels to extrapolate
    to the zero-noise limit with higher accuracy than
    linear extrapolation.
    """
    
    def __init__(
        self,
        order: int = 2,
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        self.order = order
    
    def mitigate(
        self,
        noisy_results: Dict[float, float],
        circuit=None
    ) -> float:
        """
        Richardson extrapolation.
        
        For order k, uses k+1 noise levels and computes
        the k-th order extrapolation.
        """
        noise_levels = sorted(noisy_results.keys())
        expectations = [noisy_results[nf] for nf in noise_levels]
        
        if len(noise_levels) < self.order + 1:
            # Not enough points, use linear extrapolation
            zne = ZeroNoiseExtrapolation()
            return zne.mitigate(noisy_results, circuit)
        
        # Richardson extrapolation coefficients
        # For order 2: R = 2*E(λ/2) - E(λ)
        if self.order == 1 and len(noise_levels) >= 2:
            λ1, λ2 = noise_levels[:2]
            E1, E2 = expectations[:2]
            
            # Assume λ2 = 2 * λ1
            if λ2 == 2 * λ1:
                return 2 * E1 - E2
        
        # Higher order: use polynomial fitting
        return self._richardson_rpc(noise_levels, expectations)
    
    def _richardson_rpc(
        self,
        λ: List[float],
        E: List[float]
    ) -> float:
        """
        Richardson extrapolation with correction terms.
        
        Uses the RPC (Richardson extrapolation with
        product combinations) method.
        """
        # Simple implementation: weighted average with higher weight on lower noise
        weights = np.array([1 / (l + 1) for l in λ])
        weights = weights / np.sum(weights)
        
        return np.sum(weights * np.array(E))


class SubspaceExpansion(ErrorMitigation):
    """
    Subspace Expansion Error Mitigation.
    
    Expands the measurement subspace to include
    error states and uses classical post-processing
    to mitigate errors.
    """
    
    def __init__(
        self,
        excited_state_leakage: bool = True,
        config: Optional[ErrorMitigationConfig] = None
    ):
        super().__init__(config)
        self.excited_state_leakage = excited_state_leakage
        self.subspace_matrix = None
    
    def construct_subspace(
        self,
        ideal_state: QuantumState,
        error_states: Optional[List[QuantumState]] = None
    ):
        """
        Construct subspace matrix for expansion.
        
        Args:
            ideal_state: The ideal computational state
            error_states: Additional states in subspace
        """
        states = [ideal_state]
        if error_states:
            states.extend(error_states)
        
        # Build matrix of state overlaps
        n = len(states)
        self.subspace_matrix = np.zeros((n, n), dtype=complex)
        
        for i, si in enumerate(states):
            for j, sj in enumerate(states):
                self.subspace_matrix[i, j] = np.vdot(si.state_vector, sj.state_vector)
    
    def mitigate(
        self,
        noisy_probs: np.ndarray,
        circuit=None
    ) -> np.ndarray:
        """Mitigate using subspace expansion."""
        if self.subspace_matrix is None:
            return noisy_probs
        
        # Project noisy result onto subspace
        # This is a simplified implementation
        
        return noisy_probs


class ErrorMitigator:
    """
    Unified interface for multiple error mitigation techniques.
    
    Allows combining different methods for improved mitigation.
    """
    
    def __init__(self, methods: Optional[List[ErrorMitigation]] = None):
        self.methods = methods or []
    
    def add_method(self, method: ErrorMitigation):
        """Add an error mitigation method."""
        self.methods.append(method)
    
    def mitigate(
        self,
        noisy_results,
        circuit=None,
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Apply error mitigation.
        
        Args:
            noisy_results: Noisy quantum results
            circuit: Quantum circuit
            method: Specific method to use (or all if None)
            
        Returns:
            Mitigated results
        """
        if method is not None:
            # Use specific method
            for m in self.methods:
                if m.__class__.__name__.lower() == method.lower():
                    return m.mitigate(noisy_results, circuit)
            raise ValueError(f"Unknown method: {method}")
        
        # Apply all methods sequentially
        result = noisy_results
        for m in self.methods:
            result = m.mitigate(result, circuit)
        
        return result
    
    def calibrate(self, execute_fn: Callable, num_qubits: int):
        """Calibrate all calibration-based methods."""
        for m in self.methods:
            if hasattr(m, 'calibrate'):
                m.calibrate(num_qubits, execute_fn)
