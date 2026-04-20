"""
Tests for VQC and PQC circuits.
"""

import numpy as np
import pytest
from quantumcortex.circuits import VariationalQuantumCircuit, ParameterizedQuantumCircuit


class TestVariationalQuantumCircuit:
    """Test cases for VQC."""
    
    def test_create_vqc(self):
        """Test VQC creation."""
        vqc = VariationalQuantumCircuit(
            num_qubits=4,
            num_layers=2,
            ansatz_type='hardware_efficient'
        )
        
        assert vqc.num_qubits == 4
        assert vqc.num_layers == 2
        assert len(vqc.parameters) > 0
    
    def test_forward_pass(self):
        """Test forward pass through VQC."""
        vqc = VariationalQuantumCircuit(num_qubits=4, num_layers=1)
        
        x = np.random.randn(4)
        x_normalized = np.clip(x, -1, 1) * np.pi / 2
        
        state = vqc.forward(x_normalized)
        
        assert state.num_qubits == 4
        assert np.isclose(np.linalg.norm(state.state_vector), 1.0)
    
    def test_parameter_access(self):
        """Test parameter access and modification."""
        vqc = VariationalQuantumCircuit(num_qubits=2, num_layers=1)
        
        params = vqc.get_parameters()
        assert len(params) == len(vqc.parameters)
        
        # Update parameters
        new_params = {k: np.random.uniform(0, 2*np.pi) for k in vqc.parameters.keys()}
        vqc.set_parameters(new_params)
        
        # Check update
        for k, v in vqc.parameters.items():
            assert np.isclose(v, new_params[k])
    
    def test_different_ansatz_types(self):
        """Test different ansatz types."""
        for ansatz_type in ['hardware_efficient', 'strongly_entangling', 'chemical']:
            vqc = VariationalQuantumCircuit(
                num_qubits=4,
                num_layers=2,
                ansatz_type=ansatz_type
            )
            assert vqc.ansatz_type == ansatz_type
            assert len(vqc.parameters) > 0
    
    def test_cost_function(self):
        """Test cost function evaluation."""
        vqc = VariationalQuantumCircuit(num_qubits=2, num_layers=1)
        vqc.add_observable('ZZ')
        
        x = np.array([0.5, 0.3])
        state = vqc.forward(x)
        
        cost = vqc.cost(state)
        assert isinstance(cost, float)
        assert 0 <= cost <= 2  # Z expectation in [-1, 1]


class TestParameterizedQuantumCircuit:
    """Test cases for PQC."""
    
    def test_create_pqc(self):
        """Test PQC creation."""
        pqc = ParameterizedQuantumCircuit(num_qubits=4)
        
        assert pqc.num_qubits == 4
        assert len(pqc.gates) > 0
    
    def test_add_parameter(self):
        """Test adding parameters."""
        pqc = ParameterizedQuantumCircuit(num_qubits=2)
        
        initial_count = len(pqc.parameters)
        pqc.add_parameter('test_param', initial_value=0.5)
        
        assert len(pqc.parameters) == initial_count + 1
        assert 'test_param' in pqc.parameters
    
    def test_add_layer(self):
        """Test adding layers."""
        pqc = ParameterizedQuantumCircuit(num_qubits=4)
        
        initial_gates = len(pqc.gates)
        pqc.add_layer('rotation')
        
        assert len(pqc.gates) > initial_gates
    
    def test_build_circuit(self):
        """Test circuit building."""
        pqc = ParameterizedQuantumCircuit(num_qubits=2)
        
        circuit = pqc.build_circuit()
        
        assert circuit.num_qubits == 2
        assert len(circuit.gates) > 0
    
    def test_execute_with_parameters(self):
        """Test executing with specific parameters."""
        pqc = ParameterizedQuantumCircuit(num_qubits=2)
        
        params = {k: np.random.uniform(0, 2*np.pi) for k in pqc.parameters.keys()}
        
        state = pqc.execute(params)
        
        assert state.num_qubits == 2


class TestVQCGradients:
    """Test gradient computation for VQC."""
    
    def test_compute_gradients(self):
        """Test gradient computation via parameter shift."""
        vqc = VariationalQuantumCircuit(num_qubits=2, num_layers=1)
        vqc.add_observable('ZZ')
        
        x = np.array([0.5, 0.3])
        
        gradients = vqc.compute_gradients(x, shots=100)
        
        assert isinstance(gradients, dict)
        for key, grad in gradients.items():
            assert isinstance(key, str)
            assert isinstance(grad, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
