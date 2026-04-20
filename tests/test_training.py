"""
Tests for training utilities.
"""

import numpy as np
import pytest
from quantumcortex.training import (
    GradientDescent,
    Adam,
    RMSprop,
    create_optimizer
)


class TestOptimizers:
    """Test cases for optimizers."""
    
    def test_gradient_descent(self):
        """Test gradient descent."""
        optimizer = GradientDescent(lr=0.1)
        
        params = {'w': 1.0}
        grads = {'w': 0.5}
        
        new_params = optimizer.step(params, grads)
        
        assert new_params['w'] == 1.0 - 0.1 * 0.5
    
    def test_adam(self):
        """Test Adam optimizer."""
        optimizer = Adam(lr=0.01)
        
        params = {'w': 1.0}
        grads = {'w': 0.5}
        
        new_params = optimizer.step(params, grads)
        
        assert 'w' in new_params
    
    def test_rmsprop(self):
        """Test RMSprop optimizer."""
        optimizer = RMSprop(lr=0.01)
        
        params = {'w': 1.0}
        grads = {'w': 0.5}
        
        new_params = optimizer.step(params, grads)
        
        assert 'w' in new_params
    
    def test_create_optimizer(self):
        """Test optimizer factory."""
        opt = create_optimizer('adam', lr=0.001)
        assert isinstance(opt, Adam)
        
        opt = create_optimizer('sgd', lr=0.1)
        assert isinstance(opt, GradientDescent)


class TestTrainingLoop:
    """Test cases for training loop."""
    
    def test_create_batches(self):
        """Test batch creation."""
        from quantumcortex.training.hybrid_trainer import HybridTrainer
        
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)
        
        batches = HybridTrainer._create_batches(X, y, batch_size=32)
        
        assert len(batches) > 0
        X_batch, y_batch = batches[0]
        assert X_batch.shape[0] == 32
        assert y_batch.shape[0] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
