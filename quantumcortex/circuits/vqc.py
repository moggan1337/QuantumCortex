"""
Variational Quantum Circuit (VQC) Module

Implements variational quantum circuits for quantum machine learning,
including ansätze preparation, parameterized gates, and cost function evaluation.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Union
from dataclasses import dataclass, field
import copy

from quantumcortex.core.quantum_state import QuantumState, QuantumCircuit, PAULI_I
from quantumcortex.core.measurements import ExpectationValue, ComputationalBasisMeasurement


@dataclass
class VQCLayer:
    """
    A single layer in a variational quantum circuit.
    
    Attributes:
        num_qubits: Number of qubits in this layer
        rotation_gates: List of (qubit, axis, parameter_name) for rotations
        entanglement_gates: List of (qubit1, qubit2) for entanglement
        rotation_type: Type of rotation ('rx', 'ry', 'rz', 'u3')
    """
    num_qubits: int
    rotation_gates: List[Tuple[int, str, str]] = field(default_factory=list)
    entanglement_gates: List[Tuple[int, int]] = field(default_factory=list)
    rotation_type: str = 'ry'
    
    def __post_init__(self):
        if not self.rotation_gates:
            # Default: rotate each qubit
            self.rotation_gates = [(i, self.rotation_type, f'theta_{i}') 
                                   for i in range(self.num_qubits)]
        if not self.entanglement_gates:
            # Default: CNOT chain
            self.entanglement_gates = [(i, (i + 1) % self.num_qubits) 
                                       for i in range(self.num_qubits - 1)]


class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit for quantum machine learning.
    
    A VQC consists of:
    - State preparation (encoder)
    - Parameterized ansatz layers
    - Measurement for cost function evaluation
    
    Attributes:
        num_qubits: Number of qubits
        num_layers: Number of ansatz layers
        parameters: Dictionary of parameter names to values
        ansatz_type: Type of ansatz ('hardware_efficient', 'chemical', 'strongly_entangling')
    """
    
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        ansatz_type: str = 'hardware_efficient',
        measurement_qubits: Optional[List[int]] = None,
        name: str = "VQC"
    ):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.ansatz_type = ansatz_type
        self.name = name
        
        # Measurement qubits (for reading out)
        self.measurement_qubits = measurement_qubits or list(range(num_qubits))
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Build the circuit structure
        self.layers: List[VQCLayer] = []
        self._build_layers()
        
        # Observables for cost function
        self._observables: List[np.ndarray] = []
    
    def _initialize_parameters(self):
        """Initialize circuit parameters."""
        num_params_per_layer = self.num_qubits * (1 + self.num_qubits // 2)  # rotations + entanglement
        total_params = num_params_per_layer * self.num_layers
        
        # Initialize with random values in [0, 2π]
        np.random.seed(42)
        self.parameters = {
            f'layer{l}_{i}': np.random.uniform(0, 2 * np.pi)
            for l in range(self.num_layers)
            for i in range(num_params_per_layer)
        }
        
        # Fixed parameters
        self.fixed_parameters = {}
    
    def _build_layers(self):
        """Build the VQC layer structure."""
        for layer_idx in range(self.num_layers):
            if self.ansatz_type == 'hardware_efficient':
                layer = self._build_hardware_efficient_layer(layer_idx)
            elif self.ansatz_type == 'strongly_entangling':
                layer = self._build_strongly_entangling_layer(layer_idx)
            elif self.ansatz_type == 'chemical':
                layer = self._build_chemical_layer(layer_idx)
            else:
                layer = self._build_hardware_efficient_layer(layer_idx)
            
            self.layers.append(layer)
    
    def _build_hardware_efficient_layer(self, layer_idx: int) -> VQCLayer:
        """Build hardware-efficient ansatz layer."""
        rotations = [(i, 'ry', f'l{layer_idx}_ry_{i}') for i in range(self.num_qubits)]
        rotations += [(i, 'rz', f'l{layer_idx}_rz_{i}') for i in range(self.num_qubits)]
        
        entanglements = []
        for i in range(self.num_qubits - 1):
            entanglements.append((i, i + 1))
        
        return VQCLayer(
            num_qubits=self.num_qubits,
            rotation_gates=rotations,
            entanglement_gates=entanglements,
            rotation_type='ry'
        )
    
    def _build_strongly_entangling_layer(self, layer_idx: int) -> VQCLayer:
        """Build strongly entangling layer (for 2D chip layout)."""
        rotations = [(i, 'ry', f'l{layer_idx}_ry_{i}') for i in range(self.num_qubits)]
        
        entanglements = []
        if self.num_qubits >= 2:
            entanglements.append((0, 1))
        if self.num_qubits >= 4:
            entanglements.append((2, 3))
        if self.num_qubits >= 3:
            entanglements.append((1, 2))
        
        return VQCLayer(
            num_qubits=self.num_qubits,
            rotation_gates=rotations,
            entanglement_gates=entanglements,
            rotation_type='ry'
        )
    
    def _build_chemical_layer(self, layer_idx: int) -> VQCLayer:
        """Build chemistry-inspired ansatz layer."""
        rotations = []
        for i in range(self.num_qubits):
            rotations.append((i, 'ry', f'l{layer_idx}_ry_{i}'))
            rotations.append((i, 'rz', f'l{layer_idx}_rz_{i}'))
        
        entanglements = [(i, i + 1) for i in range(self.num_qubits - 1)]
        
        return VQCLayer(
            num_qubits=self.num_qubits,
            rotation_gates=rotations,
            entanglement_gates=entanglements,
            rotation_type='ry'
        )
    
    def build_circuit(
        self,
        parameters: Optional[Dict[str, float]] = None,
        initial_state: Optional[QuantumState] = None
    ) -> QuantumCircuit:
        """
        Build the full quantum circuit with applied parameters.
        
        Args:
            parameters: Override default parameters
            initial_state: Optional initial state preparation
            
        Returns:
            QuantumCircuit ready for execution
        """
        if parameters is None:
            parameters = {**self.parameters, **self.fixed_parameters}
        
        circuit = QuantumCircuit(self.num_qubits, name=self.name)
        
        # State preparation (encoder)
        if initial_state is not None:
            # Apply encoder circuit
            self._add_encoder(circuit, initial_state)
        
        # Ansatz layers
        for layer in self.layers:
            self._add_layer(circuit, layer, parameters)
        
        return circuit
    
    def _add_encoder(self, circuit: QuantumCircuit, state: QuantumState):
        """Add encoding/initialization circuit."""
        # Simplified: just set initial state
        # Full implementation would include amplitude encoding, basis encoding, etc.
        pass
    
    def _add_layer(
        self,
        circuit: QuantumCircuit,
        layer: VQCLayer,
        parameters: Dict[str, float]
    ):
        """Add a single VQC layer to the circuit."""
        # Apply rotations
        for qubit, axis, param_name in layer.rotation_gates:
            theta = parameters.get(param_name, 0.0)
            if axis == 'rx':
                circuit.rx(qubit, theta)
            elif axis == 'ry':
                circuit.ry(qubit, theta)
            elif axis == 'rz':
                circuit.rz(qubit, theta)
        
        # Apply entangling gates
        for q1, q2 in layer.entanglement_gates:
            circuit.cnot(q1, q2)
    
    def forward(
        self,
        x: np.ndarray,
        parameters: Optional[Dict[str, float]] = None,
        observable: Optional[np.ndarray] = None
    ) -> Union[QuantumState, float]:
        """
        Forward pass through the VQC.
        
        Args:
            x: Input data (will be encoded into circuit)
            parameters: Circuit parameters
            observable: Observable to measure (if None, returns state)
            
        Returns:
            Either final state or expectation value
        """
        if parameters is None:
            parameters = self.parameters
        
        # Encode input data
        self._encode_input(x)
        
        # Build and execute circuit
        circuit = self.build_circuit(parameters)
        state = circuit.execute()
        
        if observable is not None:
            exp_val = ExpectationValue(observable)
            return exp_val(state)
        
        return state
    
    def _encode_input(self, x: np.ndarray):
        """Encode classical data into quantum circuit."""
        # Different encoding strategies:
        # 1. Basis encoding: |x⟩
        # 2. Amplitude encoding: encode in amplitudes
        # 3. Angle encoding: rotations based on x
        
        # Angle encoding for simplicity
        for i, val in enumerate(x[:self.num_qubits]):
            param_name = f'input_{i}'
            self.parameters[param_name] = float(val)
    
    def set_parameters(self, parameters: Dict[str, float]):
        """Update circuit parameters."""
        self.parameters.update(parameters)
    
    def get_parameters(self) -> np.ndarray:
        """Get all parameters as a flat array."""
        return np.array([self.parameters[k] for k in sorted(self.parameters.keys())])
    
    def set_parameters_from_array(self, params: np.ndarray):
        """Set parameters from a flat array."""
        keys = sorted(self.parameters.keys())
        for i, key in enumerate(keys):
            if i < len(params):
                self.parameters[key] = float(params[i])
    
    def add_observable(self, observable: Union[str, np.ndarray]):
        """Add an observable to the cost function."""
        if isinstance(observable, str):
            self._observables.append(self._pauli_string_to_matrix(observable))
        else:
            self._observables.append(observable)
    
    def cost(self, state: QuantumState) -> float:
        """
        Evaluate cost function (expectation value of observables).
        
        Args:
            state: Output state from circuit
            
        Returns:
            Cost value
        """
        if not self._observables:
            # Default: return negative fidelity with |0...0⟩
            target = QuantumState.zero(self.num_qubits)
            fidelity = state.fidelity(target)
            return 1.0 - fidelity
        
        total = 0.0
        for obs in self._observables:
            exp_val = ExpectationValue(obs)
            total += exp_val(state)
        
        return np.real(total) / len(self._observables)
    
    def compute_gradients(
        self,
        x: np.ndarray,
        shots: int = 1000
    ) -> Dict[str, float]:
        """
        Compute gradients using parameter shift rule.
        
        Args:
            x: Input data
            shots: Number of measurement shots
            
        Returns:
            Dictionary mapping parameter names to gradient values
        """
        gradients = {}
        h = np.pi / 2  # Shift angle
        
        for param_name, param_value in self.parameters.items():
            # Forward pass with shifted parameters
            params_plus = self.parameters.copy()
            params_plus[param_name] = param_value + h
            
            state_plus = self.forward(x, params_plus)
            cost_plus = self.cost(state_plus)
            
            # Backward pass
            params_minus = self.parameters.copy()
            params_minus[param_name] = param_value - h
            
            state_minus = self.forward(x, params_minus)
            cost_minus = self.cost(state_minus)
            
            # Parameter shift gradient
            gradients[param_name] = (cost_plus - cost_minus) / 2
        
        return gradients
    
    def reset_parameters(self, seed: Optional[int] = None):
        """Reset parameters to random values."""
        if seed is not None:
            np.random.seed(seed)
        
        for key in self.parameters:
            self.parameters[key] = np.random.uniform(0, 2 * np.pi)
    
    def copy(self) -> 'VariationalQuantumCircuit':
        """Create a deep copy of the VQC."""
        new_vqc = VariationalQuantumCircuit(
            self.num_qubits,
            self.num_layers,
            self.ansatz_type,
            self.measurement_qubits,
            self.name
        )
        new_vqc.parameters = copy.deepcopy(self.parameters)
        new_vqc.fixed_parameters = copy.deepcopy(self.fixed_parameters)
        new_vqc._observables = copy.deepcopy(self._observables)
        return new_vqc
    
    @staticmethod
    def _pauli_string_to_matrix(pauli_string: str) -> np.ndarray:
        """Convert Pauli string to matrix."""
        from quantumcortex.core.quantum_state import PAULI_X, PAULI_Y, PAULI_Z
        
        matrices = {
            'I': PAULI_I,
            'X': PAULI_X,
            'Y': PAULI_Y,
            'Z': PAULI_Z,
        }
        
        result = np.array([[1]], dtype=complex)
        for p in pauli_string:
            result = np.kron(result, matrices.get(p, PAULI_I))
        
        return result
    
    def __repr__(self) -> str:
        return (f"VariationalQuantumCircuit(num_qubits={self.num_qubits}, "
                f"num_layers={self.num_layers}, ansatz={self.ansatz_type})")
    
    def __str__(self) -> str:
        lines = [f"Variational Quantum Circuit: {self.name}"]
        lines.append(f"  Qubits: {self.num_qubits}")
        lines.append(f"  Layers: {self.num_layers}")
        lines.append(f"  Ansatz Type: {self.ansatz_type}")
        lines.append(f"  Parameters: {len(self.parameters)}")
        lines.append(f"  Observables: {len(self._observables)}")
        return "\n".join(lines)


class VQCAnsatz:
    """
    Collection of predefined VQC ansätze for different applications.
    """
    
    @staticmethod
    def hardware_efficient(num_qubits: int, num_layers: int) -> VariationalQuantumCircuit:
        """Hardware-efficient ansatz (HEA)."""
        return VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='hardware_efficient'
        )
    
    @staticmethod
    def strongly_entangling(num_qubits: int, num_layers: int) -> VariationalQuantumCircuit:
        """Strongly entangling ansatz (SEA)."""
        return VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='strongly_entangling'
        )
    
    @staticmethod
    def qaoa_like(num_qubits: int, num_layers: int) -> VariationalQuantumCircuit:
        """QAOA-inspired ansatz."""
        vqc = VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='hardware_efficient'
        )
        return vqc
    
    @staticmethod
    def chemistry_aware(num_orbitals: int, num_layers: int = 2) -> VariationalQuantumCircuit:
        """Chemistry-inspired UCCSD-like ansatz."""
        num_qubits = num_orbitals * 2  # Spin orbitals
        return VariationalQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=num_layers,
            ansatz_type='chemical'
        )
