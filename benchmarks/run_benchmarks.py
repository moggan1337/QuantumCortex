"""
Benchmarks for QuantumCortex

This module provides performance benchmarks for various
QuantumCortex components.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '..')

from quantumcortex import (
    QuantumState, QuantumCircuit,
    VariationalQuantumCircuit, ParameterizedQuantumCircuit,
    HybridQuantumClassicalModel, QNNClassifier
)


def benchmark_quantum_state_creation(n_qubits_list=[4, 6, 8, 10]):
    """Benchmark quantum state creation."""
    print("\n" + "=" * 60)
    print("Benchmark: Quantum State Creation")
    print("=" * 60)
    
    results = []
    
    for n_qubits in n_qubits_list:
        start = time.time()
        for _ in range(100):
            state = QuantumState.zero(n_qubits)
        elapsed = time.time() - start
        
        results.append({
            'n_qubits': n_qubits,
            'time_100': elapsed,
            'time_per': elapsed / 100
        })
        
        print(f"  {n_qubits} qubits: {elapsed*1000:.2f}ms for 100 states")
    
    return results


def benchmark_circuit_execution(n_qubits=4, n_gates=10):
    """Benchmark circuit execution."""
    print("\n" + "=" * 60)
    print("Benchmark: Circuit Execution")
    print("=" * 60)
    
    # Create circuit
    circuit = QuantumCircuit(num_qubits=n_qubits)
    for _ in range(n_gates):
        circuit.h(np.random.randint(0, n_qubits))
        circuit.cnot(
            np.random.randint(0, n_qubits),
            np.random.randint(0, n_qubits)
        )
    
    # Benchmark
    n_runs = 50
    start = time.time()
    for _ in range(n_runs):
        state = circuit.execute()
    elapsed = time.time() - start
    
    print(f"  Circuit: {n_qubits} qubits, {len(circuit.gates)} gates")
    print(f"  Time: {elapsed*1000:.2f}ms for {n_runs} executions")
    print(f"  Per execution: {elapsed*1000/n_runs:.2f}ms")
    
    return {
        'n_qubits': n_qubits,
        'n_gates': len(circuit.gates),
        'time_per_exec': elapsed / n_runs
    }


def benchmark_vqc_forward(n_qubits=4, n_layers=2):
    """Benchmark VQC forward pass."""
    print("\n" + "=" * 60)
    print("Benchmark: VQC Forward Pass")
    print("=" * 60)
    
    vqc = VariationalQuantumCircuit(
        num_qubits=n_qubits,
        num_layers=n_layers
    )
    
    x = np.random.randn(n_qubits) * 0.5
    
    n_runs = 100
    start = time.time()
    for _ in range(n_runs):
        state = vqc.forward(x)
    elapsed = time.time() - start
    
    print(f"  VQC: {n_qubits} qubits, {n_layers} layers")
    print(f"  Parameters: {len(vqc.parameters)}")
    print(f"  Time: {elapsed*1000:.2f}ms for {n_runs} passes")
    print(f"  Per pass: {elapsed*1000/n_runs:.2f}ms")
    
    return {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'n_params': len(vqc.parameters),
        'time_per_pass': elapsed / n_runs
    }


def benchmark_classifier_training(n_samples=100, n_features=4):
    """Benchmark classifier training."""
    print("\n" + "=" * 60)
    print("Benchmark: Classifier Training")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Create classifier
    clf = QNNClassifier(
        input_dim=n_features,
        num_classes=2,
        num_qubits=4,
        num_layers=2
    )
    
    # Train
    n_epochs = 20
    start = time.time()
    clf.fit(X, y, epochs=n_epochs, verbose=False)
    elapsed = time.time() - start
    
    print(f"  Samples: {n_samples}, Features: {n_features}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Per epoch: {elapsed/n_epochs*1000:.2f}ms")
    
    # Evaluate
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"  Training accuracy: {accuracy:.4f}")
    
    return {
        'n_samples': n_samples,
        'time_total': elapsed,
        'time_per_epoch': elapsed / n_epochs,
        'accuracy': accuracy
    }


def benchmark_hybrid_model(n_samples=50, n_features=10):
    """Benchmark hybrid model."""
    print("\n" + "=" * 60)
    print("Benchmark: Hybrid Model")
    print("=" * 60)
    
    # Create model
    model = HybridQuantumClassicalModel(
        input_dim=n_features,
        output_dim=2,
        num_qubits=4,
        num_quantum_layers=2,
        classical_pre_layers=[8],
        classical_post_layers=[4]
    )
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, 2)
    
    # Benchmark forward pass
    n_runs = 50
    start = time.time()
    for _ in range(n_runs):
        output = model.forward(X)
    elapsed = time.time() - start
    
    print(f"  Model: {n_features} input -> 2 output")
    print(f"  Samples: {n_samples}")
    print(f"  Forward pass: {elapsed*1000/n_runs:.2f}ms")
    
    return {
        'n_samples': n_samples,
        'time_per_forward': elapsed / n_runs
    }


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "#" * 60)
    print("# QuantumCortex Performance Benchmarks")
    print("#" * 60)
    
    results = {}
    
    # Run benchmarks
    results['state_creation'] = benchmark_quantum_state_creation()
    results['circuit_execution'] = benchmark_circuit_execution()
    results['vqc_forward'] = benchmark_vqc_forward()
    results['classifier_training'] = benchmark_classifier_training()
    results['hybrid_model'] = benchmark_hybrid_model()
    
    # Summary
    print("\n" + "#" * 60)
    print("# Benchmark Summary")
    print("#" * 60)
    
    print(f"\nVQC Forward Pass: {results['vqc_forward']['time_per_pass']*1000:.2f}ms")
    print(f"Classifier Training (20 epochs): {results['classifier_training']['time_total']:.2f}s")
    print(f"Hybrid Model Forward ({results['hybrid_model']['n_samples']} samples): "
          f"{results['hybrid_model']['time_per_forward']*1000:.2f}ms")
    
    print("\n" + "#" * 60)
    print("# Benchmarks Complete")
    print("#" * 60)
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()
