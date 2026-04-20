"""
Hybrid Classical-Quantum Training Module

Implements training loops for hybrid classical-quantum models
with support for mini-batching, validation, and various callbacks.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import copy

from quantumcortex.training.optimizer import (
    QuantumOptimizer, Adam, GradientDescent, create_optimizer
)


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)
    epoch_times: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 100
    batch_size: int = 32
    shuffle: bool = True
    validation_split: float = 0.0
    verbose: int = 1
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4
    lr_scheduler: Optional[str] = None
    initial_lr: float = 0.01
    gradient_clip: Optional[float] = None


class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    def on_train_begin(self, trainer: 'HybridTrainer'):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: 'HybridTrainer'):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: 'HybridTrainer'):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'HybridTrainer', logs: Dict):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, trainer: 'HybridTrainer'):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, trainer: 'HybridTrainer', logs: Dict):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to stop training when monitored metric stops improving."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, trainer: 'HybridTrainer', logs: Dict):
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.best_value is None:
            self.best_value = current
            return
        
        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True


class LearningRateScheduler(Callback):
    """Learning rate scheduling callback."""
    
    def __init__(
        self,
        schedule: Callable[[int, float], float],
        verbose: bool = True
    ):
        self.schedule = schedule
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, trainer: 'HybridTrainer'):
        new_lr = self.schedule(epoch, trainer.optimizer.lr)
        
        if new_lr != trainer.optimizer.lr:
            trainer.optimizer.lr = new_lr
            
            if self.verbose:
                print(f"Epoch {epoch}: Learning rate changed to {new_lr:.6f}")


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        verbose: int = 1
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = None
    
    def on_epoch_end(self, epoch: int, trainer: 'HybridTrainer', logs: Dict):
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        should_save = True
        
        if self.save_best_only and self.best_value is not None:
            if self.mode == 'min':
                should_save = current < self.best_value
            else:
                should_save = current > self.best_value
        
        if should_save:
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved, saving model.")
            
            self.best_value = current
            
            # Save model state
            # In practice, would serialize the model
            checkpoint = {
                'epoch': epoch,
                'model_state': trainer.model.__dict__.copy(),
                'optimizer_state': trainer.optimizer.__dict__.copy(),
                'history': trainer.history,
                'logs': logs
            }
            
            # Would save to filepath
            # np.save(self.filepath, checkpoint)


class HybridTrainer:
    """
    Trainer for hybrid classical-quantum neural networks.
    
    Handles the training loop, optimization, validation,
    and callbacks for model training.
    
    Attributes:
        model: The model to train
        optimizer: Optimizer for parameter updates
        loss_fn: Loss function
        config: Training configuration
    """
    
    def __init__(
        self,
        model: Any,
        optimizer: Optional[QuantumOptimizer] = None,
        loss_fn: Optional[Callable] = None,
        config: Optional[TrainingConfig] = None,
        metrics: Optional[List[Callable]] = None,
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model
        self.optimizer = optimizer or Adam(lr=0.001)
        self.loss_fn = loss_fn or self._default_loss
        self.config = config or TrainingConfig()
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        
        self.history = TrainingHistory()
        self.stop_training = False
        
        # Initialize optimizer with model parameters
        self._init_optimizer()
    
    def _init_optimizer(self):
        """Initialize optimizer with model parameters."""
        if hasattr(self.model, 'parameters'):
            if isinstance(self.model.parameters, dict):
                # Quantum circuit with dict parameters
                pass
            elif isinstance(self.model.parameters, np.ndarray):
                # Array parameters - wrap in dict
                pass
    
    @staticmethod
    def _default_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Default MSE loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    def _create_batches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create batches for training."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Split data into training and validation sets."""
        n_samples = len(X)
        n_val = int(n_samples * self.config.validation_split)
        
        if n_val == 0:
            return (X, y), (None, None)
        
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return (X[train_indices], y[train_indices]), (X[val_indices], y[val_indices])
    
    def _compute_gradients(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray
    ) -> Dict[str, float]:
        """Compute gradients for current batch."""
        # Forward pass
        y_pred = self.model.forward(X_batch)
        
        # Compute loss
        loss = self.loss_fn(y_batch, y_pred)
        
        # Compute gradients via parameter shift (for quantum parameters)
        if hasattr(self.model, 'compute_quantum_gradients'):
            gradients = self.model.compute_quantum_gradients(X_batch, y_batch)
        else:
            # Numerical gradients
            gradients = self._numerical_gradients(X_batch, y_batch)
        
        return gradients
    
    def _numerical_gradients(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        eps: float = 1e-5
    ) -> Dict[str, float]:
        """Compute numerical gradients (fallback)."""
        gradients = {}
        
        # Get parameters
        if hasattr(self.model, 'quantum_layer'):
            params = self.model.quantum_layer.parameters
        else:
            return gradients
        
        for name, value in params.items():
            # Forward with original params
            orig_value = value
            
            # Forward with increased param
            params[name] = value + eps
            y_pred_plus = self.model.forward(X_batch)
            loss_plus = self.loss_fn(y_batch, y_pred_plus)
            
            # Forward with decreased param
            params[name] = value - eps
            y_pred_minus = self.model.forward(X_batch)
            loss_minus = self.loss_fn(y_batch, y_pred_minus)
            
            # Gradient
            gradients[name] = (loss_plus - loss_minus) / (2 * eps)
            
            # Restore
            params[name] = orig_value
        
        return gradients
    
    def _train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, float]:
        """Train for one epoch."""
        batches = self._create_batches(
            X_train, y_train,
            self.config.batch_size,
            self.config.shuffle
        )
        
        epoch_loss = 0.0
        epoch_metrics = {m.__name__: 0.0 for m in self.metrics}
        
        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            # Callbacks
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, self)
            
            # Compute gradients
            gradients = self._compute_gradients(X_batch, y_batch)
            
            # Clip gradients if specified
            if self.config.gradient_clip is not None:
                gradients = self._clip_gradients(gradients, self.config.gradient_clip)
            
            # Update parameters
            if hasattr(self.model, 'quantum_layer'):
                params = self.model.quantum_layer.parameters
                new_params = self.optimizer.step(params, gradients)
                self.model.quantum_layer.parameters = new_params
            
            # Compute metrics
            y_pred = self.model.forward(X_batch)
            batch_loss = self.loss_fn(y_batch, y_pred)
            epoch_loss += batch_loss
            
            for metric in self.metrics:
                epoch_metrics[metric.__name__] += metric(y_batch, y_pred)
            
            # Batch callbacks
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, self, {'loss': batch_loss})
        
        # Average metrics
        n_batches = len(batches)
        epoch_loss /= n_batches
        epoch_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def _validate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Validate the model."""
        if X_val is None or y_val is None:
            return {}
        
        y_pred = self.model.forward(X_val)
        val_loss = self.loss_fn(y_val, y_pred)
        
        val_metrics = {'val_loss': val_loss}
        for metric in self.metrics:
            val_metrics[f'val_{metric.__name__}'] = metric(y_val, y_pred)
        
        return val_metrics
    
    def _clip_gradients(
        self,
        gradients: Dict[str, float],
        max_norm: float
    ) -> Dict[str, float]:
        """Clip gradients by global norm."""
        grad_values = list(gradients.values())
        total_norm = np.sqrt(sum(g ** 2 for g in grad_values))
        
        clip_factor = max_norm / (total_norm + 1e-6)
        
        if clip_factor < 1:
            return {k: g * clip_factor for k, g in gradients.items()}
        
        return gradients
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            X: Training inputs
            y: Training targets
            X_val: Optional validation inputs
            y_val: Optional validation targets
            
        Returns:
            TrainingHistory object
        """
        # Handle validation split
        if self.config.validation_split > 0 and X_val is None:
            (X_train, y_train), (X_val, y_val) = self._split_data(X, y)
        else:
            X_train, y_train = X, y
        
        # Train callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        # Training loop
        self.stop_training = False
        
        for epoch in range(self.config.epochs):
            if self.stop_training:
                break
            
            # Epoch callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)
            
            start_time = time.time()
            
            # Train epoch
            train_logs = self._train_epoch(X_train, y_train)
            
            # Validate
            val_logs = self._validate(X_val, y_val)
            
            # Combine logs
            logs = {**train_logs, **val_logs}
            
            # Update history
            self.history.train_loss.append(logs.get('loss', 0))
            self.history.val_loss.append(logs.get('val_loss', 0))
            self.history.epoch_times.append(time.time() - start_time)
            self.history.learning_rates.append(self.optimizer.lr)
            
            for key, value in logs.items():
                if key not in self.history.train_metrics:
                    self.history.train_metrics[key] = []
                if 'val_' not in key:
                    self.history.train_metrics[key].append(value)
                
                if key not in self.history.val_metrics:
                    self.history.val_metrics[key] = []
                if 'val_' in key:
                    self.history.val_metrics[key].append(value)
            
            # Epoch callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, logs)
            
            # Print progress
            if self.config.verbose > 0:
                self._print_progress(epoch, logs)
        
        # End callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.history
    
    def _print_progress(self, epoch: int, logs: Dict):
        """Print training progress."""
        if self.config.verbose == 1:
            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1}/{self.config.epochs}"
                for key, value in logs.items():
                    if not key.startswith('val_') or len(logs) == 1:
                        msg += f" - {key}: {value:.4f}"
                print(msg)
        elif self.config.verbose > 1:
            msg = f"Epoch {epoch+1}/{self.config.epochs}"
            for key, value in logs.items():
                msg += f" - {key}: {value:.4f}"
            print(msg)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test inputs
            y: Test targets
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metric values
        """
        batches = self._create_batches(X, y, batch_size, shuffle=False)
        
        total_loss = 0.0
        total_metrics = {m.__name__: 0.0 for m in self.metrics}
        
        for X_batch, y_batch in batches:
            y_pred = self.model.forward(X_batch)
            total_loss += self.loss_fn(y_batch, y_pred)
            
            for metric in self.metrics:
                total_metrics[metric.__name__] += metric(y_batch, y_pred)
        
        n_batches = len(batches)
        results = {'loss': total_loss / n_batches}
        results.update({k: v / n_batches for k, v in total_metrics.items()})
        
        return results
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        n_samples = len(X)
        predictions = []
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            pred = self.model.forward(X[start:end])
            predictions.append(pred)
        
        return np.vstack(predictions)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=-1)
    return np.mean(y_true == y_pred)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def exponential_decay(epoch: int, initial_lr: float, decay_rate: float = 0.95) -> float:
    """Exponential learning rate decay schedule."""
    return initial_lr * (decay_rate ** epoch)


def step_decay(epoch: int, initial_lr: float, drop_every: int = 10, drop_rate: float = 0.5) -> float:
    """Step learning rate decay schedule."""
    return initial_lr * (drop_rate ** (epoch // drop_every))
