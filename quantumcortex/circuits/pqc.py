"""
Parameterized Quantum Circuit (PQC) Module

Implements parameterized quantum circuits with support for
different parameterization schemes and gradient computation.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import copy

from quantumcortex.core.quantum_state import QuantumState, QuantumCircuit
from quantumcortex.core.operators import Gate, RotationGate, PauliGate


class ParameterizationType(Enum):
    """Different parameterization schemes for quantum circuits."""
    NATURAL = "natural"           # Direct θ parameters for rotations
    EXPOLAR = "expolar"           # Exponential polar coordinates
    QUANTUM_EIGENMETRICS = "qem"  # Quantum-inspired parameterization


@dataclass
class Parameter:
    """
    A trainable parameter in the circuit.
    
    Attributes:
        name: Parameter identifier
        value: Current value
        bounds: Optional (min, max) bounds
        gradient: Current gradient estimate
    """
    name: str
    value: float
    bounds: Optional[Tuple[float, float]] = None
    gradient: float = 0.0
    
    def __post_init__(self):
        if self.bounds is not None:
            self.value = np.clip(self.value, self.bounds[0], self.bounds[1])


@dataclass
class GateInstance:
    """
    A gate instance with parameters.
    
    Attributes:
        gate_type: Type of gate (e.g., 'rx', 'ry', 'rz', 'cnot')
        qubits: Target qubit(s)
        parameter_names: Names of parameters for this gate
    """
    gate_type: str
    qubits: List[int]
    parameter_names: List[str] = field(default_factory=list)
    
    def num_parameters(self) -> int:
        return len(self.parameter_names)


class ParameterizedQuantumCircuit:
    """
    Parameterized Quantum Circuit with automatic differentiation support.
    
    Supports:
    - Multiple parameterization schemes
    - Efficient gradient computation via parameter shift
    - Circuit composition and layering
    
    Attributes:
        num_qubits: Number of qubits
        name: Circuit name
    """
    
    def __init__(
        self,
        num_qubits: int,
        name: str = "PQC",
        parameterization: ParameterizationType = ParameterizationType.NATURAL
    ):
        self.num_qubits = num_qubits
        self.name = name
        self.parameterization = parameterization
        
        # Parameters
        self.parameters: Dict[str, Parameter] = {}
        self._param_order: List[str] = []  # Maintain insertion order
        
        # Gates
        self.gates: List[GateInstance] = []
        
        # Metadata
        self._gradient_cache: Dict[str, float] = {}
        self._forward_state: Optional[QuantumState] = None
        
        # Initialize with default structure
        self._initialize_default_circuit()
    
    def _initialize_default_circuit(self):
        """Initialize with a simple default structure."""
        # Add single-qubit rotations on each qubit
        for i in range(self.num_qubits):
            self.add_parameter(f'theta_{i}')
            self.add_gate('ry', [i], [f'theta_{i}'])
        
        # Add entangling layer
        for i in range(self.num_qubits - 1):
            self.add_gate('cnot', [i, i + 1])
    
    def add_parameter(
        self,
        name: str,
        initial_value: Optional[float] = None,
        bounds: Optional[Tuple[float, float]] = None
    ) -> 'ParameterizedQuantumCircuit':
        """
        Add a trainable parameter.
        
        Args:
            name: Parameter name
            initial_value: Initial value (random if None)
            bounds: Optional (min, max) bounds
            
        Returns:
            Self for chaining
        """
        if name in self.parameters:
            raise ValueError(f"Parameter {name} already exists")
        
        if initial_value is None:
            initial_value = np.random.uniform(0, 2 * np.pi)
        
        self.parameters[name] = Parameter(
            name=name,
            value=initial_value,
            bounds=bounds
        )
        self._param_order.append(name)
        
        return self
    
    def add_gate(
        self,
        gate_type: str,
        qubits: List[int],
        parameter_names: Optional[List[str]] = None
    ) -> 'ParameterizedQuantumCircuit':
        """
        Add a gate to the circuit.
        
        Args:
            gate_type: Type of gate
            qubits: Target qubit(s)
            parameter_names: Names of parameters for parameterized gates
            
        Returns:
            Self for chaining
        """
        if parameter_names is None:
            parameter_names = []
        
        # Validate parameters exist
        for pname in parameter_names:
            if pname not in self.parameters:
                # Auto-create parameter
                self.add_parameter(pname)
        
        gate_instance = GateInstance(
            gate_type=gate_type,
            qubits=qubits,
            parameter_names=parameter_names
        )
        
        self.gates.append(gate_instance)
        return self
    
    def add_layer(
        self,
        layer_type: str = 'entanglement',
        parameter_prefix: str = 'layer'
    ) -> 'ParameterizedQuantumCircuit':
        """
        Add a layer of gates.
        
        Args:
            layer_type: Type of layer
            parameter_prefix: Prefix for new parameter names
            
        Returns:
            Self for chaining
        """
        if layer_type == 'rotation':
            # Single-qubit rotations on all qubits
            for i in range(self.num_qubits):
                pname = f'{parameter_prefix}_q{i}'
                self.add_parameter(pname)
                self.add_gate('ry', [i], [pname])
        
        elif layer_type == 'entanglement':
            # CNOT chain
            for i in range(self.num_qubits - 1):
                self.add_gate('cnot', [i, i + 1])
        
        elif layer_type == 'circular_entanglement':
            # Circular CNOT pattern
            for i in range(self.num_qubits):
                self.add_gate('cnot', [i, (i + 1) % self.num_qubits])
        
        elif layer_type == 'full_entanglement':
            # All-to-all entanglement
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    self.add_gate('cnot', [i, j])
        
        elif layer_type == 'single_layer':
            # Combined rotation + entanglement
            for i in range(self.num_qubits):
                pname = f'{parameter_prefix}_rot_q{i}'
                self.add_parameter(pname)
                self.add_gate('ry', [i], [pname])
            
            for i in range(self.num_qubits - 1):
                self.add_gate('cnot', [i, i + 1])
        
        return self
    
    def build_circuit(
        self,
        parameters: Optional[Dict[str, float]] = None
    ) -> QuantumCircuit:
        """
        Build the quantum circuit with applied parameters.
        
        Args:
            parameters: Parameter values to use
            
        Returns:
            QuantumCircuit ready for execution
        """
        if parameters is None:
            parameters = {name: p.value for name, p in self.parameters.items()}
        
        circuit = QuantumCircuit(self.num_qubits, name=self.name)
        
        for gate in self.gates:
            self._apply_gate_to_circuit(circuit, gate, parameters)
        
        return circuit
    
    def _apply_gate_to_circuit(
        self,
        circuit: QuantumCircuit,
        gate: GateInstance,
        parameters: Dict[str, float]
    ):
        """Apply a gate to the circuit."""
        gate_type = gate.gate_type.lower()
        qubits = gate.qubits
        
        if gate_type == 'h':
            circuit.h(qubits[0])
        
        elif gate_type in ['rx', 'ry', 'rz']:
            theta = parameters.get(gate.parameter_names[0], 0.0) if gate.parameter_names else 0.0
            if gate_type == 'rx':
                circuit.rx(qubits[0], theta)
            elif gate_type == 'ry':
                circuit.ry(qubits[0], theta)
            else:
                circuit.rz(qubits[0], theta)
        
        elif gate_type == 'u3':
            # Universal 3-parameter gate
            thetas = [parameters.get(p, 0.0) for p in gate.parameter_names]
            theta, phi, lam = thetas if len(thetas) == 3 else (thetas[0], 0.0, 0.0)
            circuit.u3(qubits[0], theta, phi, lam)
        
        elif gate_type == 'cnot':
            circuit.cnot(qubits[0], qubits[1])
        
        elif gate_type == 'cz':
            circuit.cz(qubits[0], qubits[1])
        
        elif gate_type == 'swap':
            circuit.swap(qubits[0], qubits[1])
        
        elif gate_type == 'x':
            circuit.x(qubits[0])
        elif gate_type == 'y':
            circuit.y(qubits[0])
        elif gate_type == 'z':
            circuit.z(qubits[0])
        
        elif gate_type == 's':
            circuit.s(qubits[0])
        elif gate_type == 't':
            circuit.t(qubits[0])
    
    def execute(
        self,
        parameters: Optional[Dict[str, float]] = None,
        initial_state: Optional[QuantumState] = None
    ) -> QuantumState:
        """
        Execute the circuit.
        
        Args:
            parameters: Parameter values
            initial_state: Initial quantum state
            
        Returns:
            Final quantum state
        """
        circuit = self.build_circuit(parameters)
        self._forward_state = circuit.execute(initial_state)
        return self._forward_state
    
    def forward(
        self,
        x: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, float]] = None
    ) -> QuantumState:
        """
        Forward pass through the PQC.
        
        Args:
            x: Optional input data
            parameters: Circuit parameters
            
        Returns:
            Output quantum state
        """
        if parameters is None:
            parameters = {name: p.value for name, p in self.parameters.items()}
        
        # Encode input if provided
        if x is not None:
            parameters = self._encode_input(x, parameters)
        
        return self.execute(parameters)
    
    def _encode_input(
        self,
        x: np.ndarray,
        parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Encode classical input into parameters.
        
        Args:
            x: Input data
            parameters: Current parameters
            
        Returns:
            Updated parameters with encoded input
        """
        # Angle encoding: use input values as rotation angles
        for i, val in enumerate(x):
            if i < self.num_qubits:
                param_name = f'input_{i}'
                if param_name not in parameters:
                    self.add_parameter(param_name, float(val))
                else:
                    parameters[param_name] = float(val)
        
        return parameters
    
    def expectation_value(
        self,
        observable: Union[str, np.ndarray],
        parameters: Optional[Dict[str, float]] = None,
        shots: Optional[int] = None
    ) -> float:
        """
        Compute expectation value of an observable.
        
        Args:
            observable: Observable (Pauli string or matrix)
            parameters: Circuit parameters
            shots: If not None, use sampling with this many shots
            
        Returns:
            Expectation value
        """
        state = self.execute(parameters)
        
        if isinstance(observable, str):
            from quantumcortex.core.measurements import ExpectationValue
            exp_calc = ExpectationValue(observable)
            return exp_calc(state)
        else:
            exp_calc = ExpectationValue(observable)
            return exp_calc(state)
    
    def compute_gradient(
        self,
        param_name: str,
        cost_function: Callable,
        parameters: Optional[Dict[str, float]] = None,
        shots: int = 1000
    ) -> float:
        """
        Compute gradient for a single parameter using parameter shift.
        
        Args:
            param_name: Name of parameter to differentiate
            cost_function: Function that takes (parameters, shots) and returns cost
            parameters: Base parameters
            shots: Number of measurement shots
            
        Returns:
            Gradient value
        """
        if parameters is None:
            parameters = {name: p.value for name, p in self.parameters.items()}
        
        shift = np.pi / 2
        
        # Shifted parameters (positive)
        params_plus = parameters.copy()
        params_plus[param_name] = parameters[param_name] + shift
        cost_plus = cost_function(params_plus, shots)
        
        # Shifted parameters (negative)
        params_minus = parameters.copy()
        params_minus[param_name] = parameters[param_name] - shift
        cost_minus = cost_function(params_minus, shots)
        
        # Parameter shift rule
        gradient = (cost_plus - cost_minus) / 2
        
        return gradient
    
    def compute_gradients(
        self,
        cost_function: Callable,
        parameters: Optional[Dict[str, float]] = None,
        shots: int = 1000
    ) -> Dict[str, float]:
        """
        Compute gradients for all parameters.
        
        Args:
            cost_function: Function that takes (parameters, shots) and returns cost
            parameters: Base parameters
            shots: Number of measurement shots
            
        Returns:
            Dictionary mapping parameter names to gradients
        """
        if parameters is None:
            parameters = {name: p.value for name, p in self.parameters.items()}
        
        gradients = {}
        for param_name in self.parameters:
            gradients[param_name] = self.compute_gradient(
                param_name, cost_function, parameters, shots
            )
        
        return gradients
    
    def get_parameters(self) -> np.ndarray:
        """Get all parameters as a flat array."""
        return np.array([self.parameters[name].value for name in self._param_order])
    
    def set_parameters(self, parameters: Union[Dict[str, float], np.ndarray]):
        """Set parameters from dict or array."""
        if isinstance(parameters, np.ndarray):
            for i, name in enumerate(self._param_order):
                if i < len(parameters):
                    self.parameters[name].value = float(parameters[i])
        else:
            for name, value in parameters.items():
                if name in self.parameters:
                    self.parameters[name].value = float(value)
    
    def get_bounds(self) -> List[Optional[Tuple[float, float]]]:
        """Get bounds for all parameters."""
        return [self.parameters[name].bounds for name in self._param_order]
    
    def num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return len(self.parameters)
    
    def depth(self) -> int:
        """Estimate circuit depth."""
        if not self.gates:
            return 0
        
        # Simple depth estimation
        max_qubit_usage = [0] * self.num_qubits
        for gate in self.gates:
            for q in gate.qubits:
                max_qubit_usage[q] += 1
        
        return max(max_qubit_usage) if max_qubit_usage else 0
    
    def copy(self) -> 'ParameterizedQuantumCircuit':
        """Create a deep copy."""
        new_pqc = ParameterizedQuantumCircuit(
            self.num_qubits,
            self.name,
            self.parameterization
        )
        
        new_pqc.parameters = {
            name: Parameter(
                name=p.name,
                value=p.value,
                bounds=p.bounds,
                gradient=p.gradient
            )
            for name, p in self.parameters.items()
        }
        new_pqc._param_order = self._param_order.copy()
        new_pqc.gates = copy.deepcopy(self.gates)
        
        return new_pqc
    
    def __len__(self) -> int:
        return len(self.gates)
    
    def __repr__(self) -> str:
        return (f"ParameterizedQuantumCircuit(num_qubits={self.num_qubits}, "
                f"parameters={len(self.parameters)}, gates={len(self.gates)})")


class PQCBuilder:
    """
    Builder class for constructing parameterized quantum circuits.
    """
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.pqc = ParameterizedQuantumCircuit(num_qubits)
    
    def rotation_layer(self, prefix: str = 'rot') -> 'PQCBuilder':
        """Add single-qubit rotation layer."""
        self.pqc.add_layer('rotation', prefix)
        return self
    
    def entanglement_layer(
        self,
        pattern: str = 'chain'
    ) -> 'PQCBuilder':
        """Add entanglement layer."""
        if pattern == 'chain':
            self.pqc.add_layer('entanglement')
        elif pattern == 'circular':
            self.pqc.add_layer('circular_entanglement')
        elif pattern == 'full':
            self.pqc.add_layer('full_entanglement')
        return self
    
    def ansatz_block(
        self,
        name: str = 'block'
    ) -> 'PQCBuilder':
        """Add a full ansatz block (rotation + entanglement)."""
        self.pqc.add_layer('single_layer', name)
        return self
    
    def repeated_block(
        self,
        num_blocks: int,
        name_prefix: str = 'block'
    ) -> 'PQCBuilder':
        """Add multiple ansatz blocks."""
        for i in range(num_blocks):
            self.pqc.add_layer('single_layer', f'{name_prefix}_{i}')
        return self
    
    def custom_gate(
        self,
        gate_type: str,
        qubits: List[int],
        parameters: Optional[List[str]] = None
    ) -> 'PQCBuilder':
        """Add custom gate."""
        self.pqc.add_gate(gate_type, qubits, parameters)
        return self
    
    def build(self) -> ParameterizedQuantumCircuit:
        """Build and return the circuit."""
        return self.pqc
