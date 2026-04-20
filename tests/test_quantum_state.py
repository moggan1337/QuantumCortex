"""
Tests for QuantumState and QuantumCircuit.
"""

import numpy as np
import pytest
from quantumcortex.core.quantum_state import (
    QuantumState, QuantumCircuit, PAULI_I, PAULI_X, PAULI_Y, PAULI_Z, HADAMARD, CNOT
)


class TestQuantumState:
    """Test cases for QuantumState class."""
    
    def test_create_zero_state(self):
        """Test creation of |0⟩ state."""
        state = QuantumState.zero(num_qubits=2)
        assert state.num_qubits == 2
        assert state.state_vector.shape == (4,)
        assert np.isclose(state.state_vector[0], 1.0)
        assert np.isclose(np.linalg.norm(state.state_vector), 1.0)
    
    def test_create_plus_state(self):
        """Test creation of uniform superposition."""
        state = QuantumState.plus(num_qubits=2)
        assert state.num_qubits == 2
        assert np.isclose(np.linalg.norm(state.state_vector), 1.0)
    
    def test_random_state(self):
        """Test random state generation."""
        state = QuantumState.random(num_qubits=2, seed=42)
        assert state.num_qubits == 2
        assert np.isclose(np.linalg.norm(state.state_vector), 1.0)
    
    def test_apply_single_qubit_gate(self):
        """Test applying single-qubit gates."""
        state = QuantumState.zero(num_qubits=1)
        new_state = state.apply_gate(HADAMARD, [0])
        
        # H|0⟩ = (|0⟩ + |1⟩)/√2
        expected = np.array([1, 1]) / np.sqrt(2)
        assert np.allclose(new_state.state_vector, expected)
    
    def test_apply_cnot(self):
        """Test CNOT gate."""
        # |00⟩ -> |00⟩
        state = QuantumState.zero(num_qubits=2)
        new_state = state.apply_two_qubit_gate(CNOT, 0, 1)
        
        # Should remain |00⟩
        assert np.isclose(np.abs(new_state.state_vector[0]), 1.0)
    
    def test_measurement(self):
        """Test measurement sampling."""
        state = QuantumState.plus(num_qubits=1)
        counts = state.measure(shots=1000)
        
        # Should be roughly 50/50
        assert '0' in counts or '1' in counts
        for outcome, prob in counts.items():
            assert 0 <= prob <= 1
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        state = QuantumState.plus(num_qubits=1)
        exp_val = state.measure_expectation(PAULI_Z)
        
        # ⟨+|Z|+⟩ = 0
        assert np.isclose(np.real(exp_val), 0.0, atol=0.1)


class TestQuantumCircuit:
    """Test cases for QuantumCircuit class."""
    
    def test_create_circuit(self):
        """Test circuit creation."""
        circuit = QuantumCircuit(num_qubits=3, name="Test")
        assert circuit.num_qubits == 3
        assert len(circuit) == 0
    
    def test_add_single_qubit_gates(self):
        """Test adding single-qubit gates."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.h(0)
        circuit.x(1)
        circuit.y(0)
        circuit.z(1)
        
        assert len(circuit) == 4
    
    def test_rotation_gates(self):
        """Test rotation gates."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.rx(0, np.pi/4)
        circuit.ry(1, np.pi/3)
        circuit.rz(0, np.pi/2)
        
        assert len(circuit) == 3
    
    def test_two_qubit_gates(self):
        """Test two-qubit gates."""
        circuit = QuantumCircuit(num_qubits=3)
        circuit.cnot(0, 1)
        circuit.cz(1, 2)
        circuit.swap(0, 2)
        
        assert len(circuit) == 3
    
    def test_execute_bell_state(self):
        """Test executing Bell state circuit."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        state = circuit.execute()
        
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        assert np.isclose(np.abs(state.state_vector[0]), 1/np.sqrt(2))
        assert np.isclose(np.abs(state.state_vector[3]), 1/np.sqrt(2))
        assert np.isclose(np.abs(state.state_vector[1]), 0.0)
        assert np.isclose(np.abs(state.state_vector[2]), 0.0)
    
    def test_circuit_depth(self):
        """Test circuit depth calculation."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.h(0)
        circuit.h(1)  # Same layer as h(0)
        circuit.cnot(0, 1)  # New layer
        
        assert circuit.depth() >= 2
    
    def test_gate_count(self):
        """Test gate counting."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.h(0)
        circuit.h(1)
        circuit.cnot(0, 1)
        
        counts = circuit.gate_count()
        assert counts['h'] == 2
        assert counts['cnot'] == 1


class TestCircuitOperations:
    """Integration tests for circuit operations."""
    
    def test_ghz_state(self):
        """Test GHZ state preparation."""
        n = 3
        circuit = QuantumCircuit(num_qubits=n)
        
        circuit.h(0)
        for i in range(n - 1):
            circuit.cnot(i, i + 1)
        
        state = circuit.execute()
        
        # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        assert np.isclose(np.abs(state.state_vector[0]), 1/np.sqrt(2))
        assert np.isclose(np.abs(state.state_vector[-1]), 1/np.sqrt(2))
    
    def test_unitary_matrix(self):
        """Test unitary matrix extraction."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.h(0)
        
        U = circuit.get_unitary()
        
        # H should be unitary
        assert np.allclose(U @ np.conj(U.T), np.eye(2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
