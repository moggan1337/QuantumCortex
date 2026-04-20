"""
Quantum Neural Network Optimizers

Implements gradient-based optimizers for training
variational quantum circuits.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

from quantumcortex.circuits.vqc import VariationalQuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit


class QuantumOptimizer(ABC):
    """
    Abstract base class for quantum circuit optimizers.
    
    Optimizers adjust circuit parameters to minimize
    a cost function.
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        regularization: float = 0.0
    ):
        self.lr = lr
        self.regularization = regularization
        self.iteration = 0
    
    @abstractmethod
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Perform one optimization step.
        
        Args:
            parameters: Current parameter values
            gradients: Computed gradients
            
        Returns:
            Updated parameters
        """
        pass
    
    def regularize(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply L2 regularization to gradients."""
        if self.regularization == 0:
            return gradients
        
        return {
            k: g + self.regularization * p
            for k, g, p in zip(
                parameters.keys(),
                gradients.values(),
                parameters.values()
            )
        }
    
    def zero_grad(self):
        """Reset optimizer state."""
        self.iteration = 0


class GradientDescent(QuantumOptimizer):
    """
    Vanilla Gradient Descent Optimizer.
    
    θ_{t+1} = θ_t - lr * ∇C(θ_t)
    """
    
    def __init__(self, lr: float = 0.01, regularization: float = 0.0):
        super().__init__(lr, regularization)
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform gradient descent step."""
        self.iteration += 1
        
        new_params = {}
        for key in parameters:
            grad = gradients.get(key, 0.0)
            new_params[key] = parameters[key] - self.lr * grad
        
        return new_params


class Momentum(QuantumOptimizer):
    """
    Gradient Descent with Momentum.
    
    v_{t+1} = β * v_t + (1 - β) * ∇C(θ_t)
    θ_{t+1} = θ_t - lr * v_{t+1}
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        regularization: float = 0.0
    ):
        super().__init__(lr, regularization)
        self.momentum = momentum
        self._velocity: Dict[str, float] = {}
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform momentum update."""
        self.iteration += 1
        
        new_params = {}
        new_velocity = {}
        
        for key in parameters:
            # Initialize velocity
            if key not in self._velocity:
                self._velocity[key] = 0.0
            
            # Update velocity
            self._velocity[key] = (
                self.momentum * self._velocity[key] +
                (1 - self.momentum) * gradients.get(key, 0.0)
            )
            
            # Update parameters
            new_params[key] = parameters[key] - self.lr * self._velocity[key]
        
        self._velocity = new_velocity
        return new_params
    
    def zero_grad(self):
        """Reset velocity."""
        super().zero_grad()
        self._velocity = {}


class Adam(QuantumOptimizer):
    """
    Adam (Adaptive Moment Estimation) Optimizer.
    
    Widely used optimizer with adaptive learning rates
    and momentum-like behavior.
    
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t       (first moment)
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²      (second moment)
    m_hat = m_t / (1 - β₁^t)
    v_hat = v_t / (1 - β₂^t)
    θ_{t+1} = θ_t - lr * m_hat / (√v_hat + ε)
    """
    
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        regularization: float = 0.0
    ):
        super().__init__(lr, regularization)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self._m: Dict[str, float] = {}
        self._v: Dict[str, float] = {}
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform Adam update."""
        self.iteration += 1
        
        # Bias correction
        beta1_t = self.beta1 ** self.iteration
        beta2_t = self.beta2 ** self.iteration
        
        new_params = {}
        new_m = {}
        new_v = {}
        
        for key in parameters:
            # Initialize moments
            if key not in self._m:
                self._m[key] = 0.0
                self._v[key] = 0.0
            
            grad = gradients.get(key, 0.0)
            
            # Update biased first moment estimate
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * grad ** 2
            
            # Compute bias-corrected estimates
            m_hat = self._m[key] / (1 - beta1_t)
            v_hat = self._v[key] / (1 - beta2_t)
            
            # Update parameters
            new_params[key] = parameters[key] - self.lr * m_hat / (
                np.sqrt(v_hat) + self.epsilon
            )
            
            new_m[key] = self._m[key]
            new_v[key] = self._v[key]
        
        self._m = new_m
        self._v = new_v
        return new_params
    
    def zero_grad(self):
        """Reset moments."""
        super().zero_grad()
        self._m = {}
        self._v = {}


class RMSprop(QuantumOptimizer):
    """
    RMSprop Optimizer.
    
    Uses adaptive learning rates based on gradient magnitudes.
    
    E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g²
    θ_{t+1} = θ_t - lr * g_t / √(E[g²]_t + ε)
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        regularization: float = 0.0
    ):
        super().__init__(lr, regularization)
        self.rho = rho
        self.epsilon = epsilon
        
        self._E_g2: Dict[str, float] = {}
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform RMSprop update."""
        self.iteration += 1
        
        new_params = {}
        new_E_g2 = {}
        
        for key in parameters:
            if key not in self._E_g2:
                self._E_g2[key] = 0.0
            
            grad = gradients.get(key, 0.0)
            
            # Update running average of squared gradients
            self._E_g2[key] = self.rho * self._E_g2[key] + (1 - self.rho) * grad ** 2
            
            # Update parameters
            new_params[key] = parameters[key] - self.lr * grad / (
                np.sqrt(self._E_g2[key]) + self.epsilon
            )
            
            new_E_g2[key] = self._E_g2[key]
        
        self._E_g2 = new_E_g2
        return new_params
    
    def zero_grad(self):
        """Reset running averages."""
        super().zero_grad()
        self._E_g2 = {}


class Adagrad(QuantumOptimizer):
    """
    Adagrad Optimizer.
    
    Adapts learning rate based on parameter history.
    Good for sparse gradients.
    
    G_t = G_{t-1} + g²_t
    θ_{t+1} = θ_t - lr * g_t / √(G_t + ε)
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        epsilon: float = 1e-8,
        regularization: float = 0.0
    ):
        super().__init__(lr, regularization)
        self.epsilon = epsilon
        
        self._G: Dict[str, float] = {}
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform Adagrad update."""
        self.iteration += 1
        
        new_params = {}
        new_G = {}
        
        for key in parameters:
            if key not in self._G:
                self._G[key] = 0.0
            
            grad = gradients.get(key, 0.0)
            
            # Accumulate squared gradients
            self._G[key] = self._G[key] + grad ** 2
            
            # Update parameters
            new_params[key] = parameters[key] - self.lr * grad / (
                np.sqrt(self._G[key]) + self.epsilon
            )
            
            new_G[key] = self._G[key]
        
        self._G = new_G
        return new_params
    
    def zero_grad(self):
        """Reset accumulated gradients."""
        super().zero_grad()
        self._G = {}


class NesterovMomentum(QuantumOptimizer):
    """
    Nesterov Accelerated Gradient (NAG) Optimizer.
    
    Looks ahead to where parameters are going to be
    for a more informed update.
    
    v_{t+1} = β * v_t + (1 - β) * ∇C(θ_t - β * v_t)
    θ_{t+1} = θ_t - lr * v_{t+1}
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        regularization: float = 0.0
    ):
        super().__init__(lr, regularization)
        self.momentum = momentum
        self._velocity: Dict[str, float] = {}
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform Nesterov accelerated gradient step."""
        self.iteration += 1
        
        new_params = {}
        new_velocity = {}
        
        for key in parameters:
            if key not in self._velocity:
                self._velocity[key] = 0.0
            
            # Nesterov look-ahead
            look_ahead = parameters[key] - self.momentum * self._velocity[key]
            
            # Update velocity with looked-ahead gradient
            self._velocity[key] = (
                self.momentum * self._velocity[key] +
                gradients.get(key, 0.0)
            )
            
            # Update parameters
            new_params[key] = look_ahead - self.lr * self._velocity[key]
            
            new_velocity[key] = self._velocity[key]
        
        self._velocity = new_velocity
        return new_params
    
    def zero_grad(self):
        """Reset velocity."""
        super().zero_grad()
        self._velocity = {}


class QuantumNaturalGradient(QuantumOptimizer):
    """
    Quantum Natural Gradient Optimizer.
    
    Uses the Fubini-Study metric tensor (quantum Fisher information)
    for parameter updates.
    
    θ_{t+1} = θ_t - lr * F⁻¹ * ∇C
    
    where F is the quantum Fisher information matrix.
    """
    
    def __init__(
        self,
        lr: float = 0.01,
        epsilon: float = 1e-6,
        regularization: float = 0.0
    ):
        super().__init__(lr, regularization)
        self.epsilon = epsilon
        self._qfi_estimate: Optional[np.ndarray] = None
    
    def compute_qfi(
        self,
        circuit: ParameterizedQuantumCircuit,
        parameters: Dict[str, float],
        state
    ) -> np.ndarray:
        """
        Estimate Quantum Fisher Information matrix.
        
        F_ij = 4 * Re(⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩)
        """
        # Simplified QFI computation
        # Full implementation would compute analytic gradients
        
        n_params = len(parameters)
        qfi = np.eye(n_params) * 0.5
        
        return qfi
    
    def step(
        self,
        parameters: Dict[str, float],
        gradients: Dict[str, float],
        circuit: Optional[ParameterizedQuantumCircuit] = None,
        state=None
    ) -> Dict[str, float]:
        """Perform quantum natural gradient step."""
        self.iteration += 1
        
        if circuit is not None and state is not None:
            qfi = self.compute_qfi(circuit, parameters, state)
            
            # Regularize QFI for invertibility
            qfi += self.epsilon * np.eye(len(qfi))
            
            # Compute natural gradient
            grad_array = np.array([gradients.get(k, 0.0) for k in sorted(parameters.keys())])
            natural_grad = np.linalg.solve(qfi, grad_array)
            
            # Update parameters
            keys = sorted(parameters.keys())
            new_params = {}
            for i, key in enumerate(keys):
                new_params[key] = parameters[key] - self.lr * natural_grad[i]
            
            return new_params
        else:
            # Fall back to vanilla gradient descent
            return GradientDescent(self.lr).step(parameters, gradients)


class ParameterShiftOptimizer:
    """
    Optimizer specifically designed for VQC training
    using the parameter shift rule.
    
    Uses analytic gradients computed via the parameter shift
    rule for more accurate gradient estimates.
    """
    
    def __init__(
        self,
        optimizer: QuantumOptimizer = None,
        shift: float = np.pi / 2
    ):
        self.optimizer = optimizer or Adam(lr=0.01)
        self.shift = shift
    
    def compute_gradients(
        self,
        circuit: ParameterizedQuantumCircuit,
        cost_fn: Callable,
        parameters: Dict[str, float],
        shots: int = 1000
    ) -> Dict[str, float]:
        """
        Compute gradients using parameter shift rule.
        
        ∂C/∂θ_i = (C(θ_i + π/2) - C(θ_i - π/2)) / 2
        """
        gradients = {}
        
        for param_name in parameters:
            # Shifted parameters
            params_plus = parameters.copy()
            params_plus[param_name] = parameters[param_name] + self.shift
            
            params_minus = parameters.copy()
            params_minus[param_name] = parameters[param_name] - self.shift
            
            # Evaluate cost at shifted parameters
            cost_plus = cost_fn(params_plus, shots)
            cost_minus = cost_fn(params_minus, shots)
            
            # Parameter shift gradient
            gradients[param_name] = (cost_plus - cost_minus) / 2
        
        return gradients
    
    def step(
        self,
        circuit: ParameterizedQuantumCircuit,
        cost_fn: Callable,
        parameters: Dict[str, float],
        shots: int = 1000
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform one optimization step.
        
        Returns:
            Tuple of (updated_parameters, gradients)
        """
        # Compute gradients
        gradients = self.compute_gradients(circuit, cost_fn, parameters, shots)
        
        # Update parameters
        new_parameters = self.optimizer.step(parameters, gradients)
        
        return new_parameters, gradients


def create_optimizer(
    name: str,
    lr: float = 0.01,
    **kwargs
) -> QuantumOptimizer:
    """
    Factory function to create optimizers.
    
    Args:
        name: Optimizer name ('sgd', 'adam', 'rmsprop', 'adagrad', 'momentum')
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        QuantumOptimizer instance
    """
    optimizers = {
        'sgd': GradientDescent,
        'gd': GradientDescent,
        'momentum': Momentum,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'nesterov': NesterovMomentum,
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](lr=lr, **kwargs)
