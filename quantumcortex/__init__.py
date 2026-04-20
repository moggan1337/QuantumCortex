"""
QuantumCortex - Quantum Neural Network Framework

A comprehensive framework for building and training quantum neural networks,
including variational quantum circuits, quantum perceptrons, and hybrid
classical-quantum models.
"""

__version__ = "0.1.0"
__author__ = "QuantumCortex Team"

from quantumcortex.core.quantum_state import QuantumState, QuantumCircuit
from quantumcortex.core.operators import QuantumOperator, Gate, PauliGate, RotationGate
from quantumcortex.core.measurements import Measurement, ExpectationValue
from quantumcortex.circuits.vqc import VariationalQuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit
from quantumcortex.layers.quantum_perceptron import QuantumPerceptron
from quantumcortex.layers.quantum_conv import QuantumConvolutionalLayer
from quantumcortex.layers.quantum_recurrent import QuantumRecurrentLayer
from quantumcortex.models.hybrid_model import HybridQuantumClassicalModel
from quantumcortex.training.optimizer import QuantumOptimizer, GradientDescent
from quantumcortex.training.hybrid_trainer import HybridTrainer
from quantumcortex.utils.entanglement import EntanglementAnalyzer
from quantumcortex.utils.kernel import QuantumKernel

__all__ = [
    "QuantumState",
    "QuantumCircuit",
    "QuantumOperator",
    "Gate",
    "PauliGate",
    "RotationGate",
    "Measurement",
    "ExpectationValue",
    "VariationalQuantumCircuit",
    "ParameterizedQuantumCircuit",
    "QuantumPerceptron",
    "QuantumConvolutionalLayer",
    "QuantumRecurrentLayer",
    "HybridQuantumClassicalModel",
    "QuantumOptimizer",
    "GradientDescent",
    "HybridTrainer",
    "EntanglementAnalyzer",
    "QuantumKernel",
]
