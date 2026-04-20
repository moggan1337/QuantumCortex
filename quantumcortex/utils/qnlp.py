"""
Quantum Natural Language Processing (QNLP) Module

Implements quantum approaches to natural language processing
using categorical quantum mechanics and DisCoCat (Distributional
Compositional Categorical) semantics.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

from quantumcortex.core.quantum_state import QuantumState, QuantumCircuit
from quantumcortex.circuits.pqc import ParameterizedQuantumCircuit


@dataclass
class WordMeaning:
    """Represents the meaning of a word as a quantum state."""
    word: str
    vector: np.ndarray  # Word embedding vector
    grammatical_type: str  # noun, verb, adjective, etc.
    quantum_state: Optional[QuantumState] = None


@dataclass
class GrammarRule:
    """Represents a grammatical rule for word combination."""
    name: str
    input_types: List[str]
    output_type: str
    combination_method: str  # 'tensor', ' braid', 'cup'


class QNLPEncoder:
    """
    QNLP Encoder using categorical compositional semantics.
    
    Encodes sentences into quantum states using the DisCoCat
    (Distributional Compositional Categorical) framework.
    """
    
    def __init__(
        self,
        vocabulary: Dict[str, WordMeaning],
        grammar: Dict[str, GrammarRule],
        num_qubits: int = 4
    ):
        self.vocabulary = vocabulary
        self.grammar = grammar
        self.num_qubits = num_qubits
        
        # Build type lattice
        self.type_lattice = self._build_type_lattice()
    
    def _build_type_lattice(self) -> Dict[str, str]:
        """Build the type lattice for grammar."""
        return {
            'n': 'noun',
            's': 'sentence',
            'np': 'noun_phrase',
            'vp': 'verb_phrase',
            'adj': 'adjective',
        }
    
    def encode_word(self, word: str) -> QuantumState:
        """
        Encode a single word into a quantum state.
        
        Args:
            word: The word to encode
            
        Returns:
            QuantumState representing the word's meaning
        """
        if word not in self.vocabulary:
            # Unknown word: use random state
            return QuantumState.random(self.num_qubits)
        
        word_meaning = self.vocabulary[word]
        
        if word_meaning.quantum_state is not None:
            return word_meaning.quantum_state
        
        # Encode word vector into quantum state
        return self._vector_to_state(word_meaning.vector)
    
    def _vector_to_state(self, vector: np.ndarray) -> QuantumState:
        """Convert word embedding vector to quantum state."""
        # Normalize vector
        vec_norm = vector / (np.linalg.norm(vector) + 1e-8)
        
        # Pad or truncate to match qubit count
        size = 2 ** self.num_qubits
        if len(vec_norm) < size:
            amplitudes = np.zeros(size, dtype=complex)
            amplitudes[:len(vec_norm)] = vec_norm
        else:
            amplitudes = vec_norm[:size]
        
        # Normalize amplitudes
        amplitudes = amplitudes / (np.linalg.norm(amplitudes) + 1e-8)
        
        return QuantumState(amplitudes, self.num_qubits)
    
    def encode_sentence(self, sentence: str) -> QuantumState:
        """
        Encode a complete sentence into a quantum state.
        
        Args:
            sentence: The sentence to encode
            
        Returns:
            QuantumState representing the sentence's meaning
        """
        words = sentence.lower().split()
        
        if not words:
            return QuantumState.zero(self.num_qubits)
        
        # Encode each word
        word_states = [self.encode_word(w) for w in words]
        
        # Compose states according to grammar
        # Simplified: tensor product of word states
        return self._tensor_compose(word_states)
    
    def _tensor_compose(self, states: List[QuantumState]) -> QuantumState:
        """Compose quantum states using tensor product."""
        if len(states) == 1:
            return states[0]
        
        # Tensor product of states
        combined_vector = states[0].state_vector
        for state in states[1:]:
            combined_vector = np.kron(combined_vector, state.state_vector)
        
        # Truncate to manageable size
        max_size = 2 ** self.num_qubits
        combined_vector = combined_vector[:max_size]
        
        # Normalize
        combined_vector = combined_vector / (np.linalg.norm(combined_vector) + 1e-8)
        
        return QuantumState(combined_vector, self.num_qubits)
    
    def braid_compose(
        self,
        state1: QuantumState,
        state2: QuantumState,
        qubit1: int = 0,
        qubit2: int = 1
    ) -> QuantumState:
        """
        Compose states using braiding (swap operations).
        
        Args:
            state1: First quantum state
            state2: Second quantum state
            qubit1: Qubit to braid
            qubit2: Qubit to braid with
            
        Returns:
            Braided quantum state
        """
        circuit = QuantumCircuit(state1.num_qubits)
        circuit.swap(qubit1, qubit2)
        
        return circuit.execute(state1)
    
    def apply_grammar_rule(
        self,
        states: List[QuantumState],
        rule_name: str
    ) -> QuantumState:
        """
        Apply a grammatical rule to compose states.
        
        Args:
            states: List of quantum states
            rule_name: Name of grammar rule
            
        Returns:
            Resulting composed state
        """
        if rule_name not in self.grammar:
            return self._tensor_compose(states)
        
        rule = self.grammar[rule_name]
        
        if rule.combination_method == 'tensor':
            return self._tensor_compose(states)
        elif rule.combination_method == 'braid':
            if len(states) >= 2:
                return self.braid_compose(states[0], states[1])
            return states[0] if states else QuantumState.zero(self.num_qubits)
        elif rule.combination_method == 'cup':
            # Cup (trace) operation - compute partial trace
            if states:
                return states[0]  # Simplified
            return QuantumState.zero(self.num_qubits)
        
        return self._tensor_compose(states)


class CategoricalGrammar:
    """
    Categorical Grammar for QNLP.
    
    Implements grammar as operations in a monoidal category,
    following the DisCoCat framework.
    """
    
    def __init__(self):
        self.types: Set[str] = set()
        self.rules: Dict[str, GrammarRule] = {}
    
    def add_type(self, type_name: str):
        """Add a grammatical type."""
        self.types.add(type_name)
    
    def add_rule(
        self,
        name: str,
        input_types: List[str],
        output_type: str,
        method: str = 'tensor'
    ):
        """Add a grammatical rule."""
        rule = GrammarRule(
            name=name,
            input_types=input_types,
            output_type=output_type,
            combination_method=method
        )
        self.rules[name] = rule
        self.types.update(input_types)
        self.types.add(output_type)
    
    def parse(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Parse sentence into type-annotated tokens.
        
        Args:
            sentence: Sentence to parse
            
        Returns:
            List of (word, type) tuples
        """
        # Simplified parsing
        words = sentence.lower().split()
        
        parsed = []
        for word in words:
            word_type = self._guess_type(word)
            parsed.append((word, word_type))
        
        return parsed
    
    def _guess_type(self, word: str) -> str:
        """Guess grammatical type from word ending."""
        if word.endswith(('ly', 'ful', 'less')):
            return 'adj'
        elif word.endswith(('ing', 'ed', 's')):
            return 'v'
        elif word.endswith(('tion', 'ness', 'ment')):
            return 'n'
        return 'n'  # Default to noun


class QuantumSentenceEncoder:
    """
    Parametrized Quantum Sentence Encoder.
    
    Uses a parameterized quantum circuit to encode
    sentences with learnable parameters.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_qubits: int = 4,
        num_layers: int = 2
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
        # Word embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Quantum encoder circuit
        self.encoder_circuit = self._build_encoder_circuit()
    
    def _build_encoder_circuit(self) -> ParameterizedQuantumCircuit:
        """Build the encoder quantum circuit."""
        circuit = ParameterizedQuantumCircuit(
            num_qubits=self.num_qubits,
            num_layers=self.num_layers
        )
        
        # Add encoding layers
        for layer in range(self.num_layers):
            # Embedding rotations
            for q in range(min(self.num_qubits, self.embedding_dim)):
                circuit.add_parameter(f'l{layer}_q{q}')
                circuit.add_gate('ry', [q], [f'l{layer}_q{q}'])
            
            # Entangling
            for q in range(self.num_qubits - 1):
                circuit.add_gate('cnot', [q, q + 1])
        
        return circuit
    
    def encode(
        self,
        word_ids: np.ndarray,
        parameters: Optional[Dict[str, float]] = None
    ) -> QuantumState:
        """
        Encode word sequence into quantum state.
        
        Args:
            word_ids: Word indices
            parameters: Circuit parameters
            
        Returns:
            Encoded quantum state
        """
        # Get embeddings
        embeddings = self.embeddings[word_ids]
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        # Encode into circuit
        if parameters is None:
            parameters = {}
            for name, param in self.encoder_circuit.parameters.items():
                parameters[name] = param.value
        
        # Apply to circuit
        self.encoder_circuit.set_parameters(parameters)
        
        return self.encoder_circuit.forward(avg_embedding)
    
    def forward(
        self,
        word_ids: np.ndarray,
        parameters: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Forward pass returning features."""
        state = self.encode(word_ids, parameters)
        probs = np.abs(state.state_vector) ** 2
        return probs[:self.embedding_dim]


class QuantumTextClassifier:
    """
    Quantum Text Classifier.
    
    Uses QNLP techniques for text classification tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 8,
        num_qubits: int = 4
    ):
        self.num_classes = num_classes
        
        # Sentence encoder
        self.encoder = QuantumSentenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_qubits=num_qubits
        )
        
        # Classification weights
        self.class_weights = np.random.randn(embedding_dim, num_classes) * 0.01
        self.class_bias = np.zeros(num_classes)
    
    def forward(self, word_ids: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Encode sentence
        features = self.encoder.forward(word_ids)
        
        # Classify
        logits = features @ self.class_weights + self.class_bias
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def predict(self, word_ids: np.ndarray) -> int:
        """Predict class label."""
        probs = self.forward(word_ids)
        return np.argmax(probs)
    
    def fit(
        self,
        X: List[np.ndarray],
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01
    ):
        """Train the classifier."""
        for epoch in range(epochs):
            total_loss = 0
            
            for ids, label in zip(X, y):
                # Forward pass
                probs = self.forward(ids)
                
                # Cross-entropy loss
                loss = -np.log(probs[int(label)] + 1e-10)
                total_loss += loss
                
                # Gradients (simplified)
                grad_weights = np.outer(probs, probs)
                grad_weights[int(label), :] -= probs
                
                self.class_weights -= lr * grad_weights
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(X):.4f}")


class QuantumWordEmbeddings:
    """
    Quantum Word Embeddings.
    
    Creates word embeddings using quantum circuits.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_qubits: int = 4
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_qubits = num_qubits
        
        # Embedding matrix
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Quantum transformation
        self.quantum_transform = ParameterizedQuantumCircuit(
            num_qubits=num_qubits,
            num_layers=1
        )
    
    def encode_word(self, word_id: int) -> QuantumState:
        """Encode a word into quantum state."""
        embedding = self.embeddings[word_id]
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Pad to match qubits
        size = 2 ** self.num_qubits
        amplitudes = np.zeros(size, dtype=complex)
        amplitudes[:len(embedding)] = embedding
        
        return QuantumState(amplitudes, self.num_qubits)
    
    def compute_similarity(
        self,
        word1_id: int,
        word2_id: int
    ) -> float:
        """Compute quantum similarity between words."""
        state1 = self.encode_word(word1_id)
        state2 = self.encode_word(word2_id)
        
        # Fidelity
        return state1.fidelity(state2)
    
    def most_similar(
        self,
        word_id: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most similar words."""
        state = self.encode_word(word_id)
        
        similarities = []
        for i in range(self.vocab_size):
            if i != word_id:
                state_i = self.encode_word(i)
                sim = state.fidelity(state_i)
                similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class QNLPDemo:
    """
    Demonstration of QNLP capabilities.
    """
    
    def __init__(self):
        self.encoder = None
        self.grammar = None
    
    def setup_simple_vocabulary(self) -> Dict[str, WordMeaning]:
        """Set up a simple vocabulary for demonstration."""
        vocabulary = {}
        
        # Simple word embeddings (random for demo)
        np.random.seed(42)
        
        words_data = [
            ('cat', 'n', np.array([1.0, 0.0, 0.0, 0.0])),
            ('dog', 'n', np.array([0.0, 1.0, 0.0, 0.0])),
            ('run', 'v', np.array([0.0, 0.0, 1.0, 0.0])),
            ('big', 'adj', np.array([0.0, 0.0, 0.0, 1.0])),
            ('small', 'adj', np.array([0.5, 0.5, 0.0, 0.0])),
            ('chase', 'v', np.array([0.3, 0.7, 0.0, 0.0])),
        ]
        
        for word, gtype, vec in words_data:
            meaning = WordMeaning(
                word=word,
                vector=vec,
                grammatical_type=gtype
            )
            vocabulary[word] = meaning
        
        return vocabulary
    
    def setup_grammar(self) -> Dict[str, GrammarRule]:
        """Set up grammatical rules."""
        grammar = {
            'subject_verb': GrammarRule(
                name='subject_verb',
                input_types=['n', 'v'],
                output_type='vp',
                combination_method='tensor'
            ),
            'adj_noun': GrammarRule(
                name='adj_noun',
                input_types=['adj', 'n'],
                output_type='n',
                combination_method='tensor'
            ),
            'np_vp': GrammarRule(
                name='np_vp',
                input_types=['np', 'vp'],
                output_type='s',
                combination_method='tensor'
            ),
        }
        
        return grammar
    
    def demo_sentence_encoding(self):
        """Demonstrate sentence encoding."""
        vocabulary = self.setup_simple_vocabulary()
        grammar = self.setup_grammar()
        
        encoder = QNLPEncoder(
            vocabulary=vocabulary,
            grammar=grammar,
            num_qubits=4
        )
        
        # Encode sentences
        sentences = [
            "big cat",
            "small dog",
            "cat run",
            "dog chase",
        ]
        
        print("QNLP Sentence Encoding Demo")
        print("=" * 50)
        
        for sentence in sentences:
            state = encoder.encode_sentence(sentence)
            print(f"\nSentence: '{sentence}'")
            print(f"  Number of qubits: {state.num_qubits}")
            print(f"  State vector norm: {np.linalg.norm(state.state_vector):.4f}")
        
        # Compare meanings
        print("\n" + "=" * 50)
        print("Meaning Similarities:")
        
        cat_state = encoder.encode_sentence("cat")
        dog_state = encoder.encode_sentence("dog")
        
        similarity = cat_state.fidelity(dog_state)
        print(f"  Similarity(cat, dog): {similarity:.4f}")
    
    def demo_text_classification(self):
        """Demonstrate text classification."""
        print("\n" + "=" * 50)
        print("QNLP Text Classification Demo")
        print("=" * 50)
        
        # Create simple vocabulary
        vocab = {'the': 0, 'cat': 1, 'dog': 2, 'runs': 3, 'sleeps': 4, '0': 5, '1': 6}
        
        classifier = QuantumTextClassifier(
            vocab_size=len(vocab),
            num_classes=2,
            embedding_dim=8,
            num_qubits=4
        )
        
        # Simple training data
        X_train = [
            np.array([0, 1, 3]),  # "the cat runs"
            np.array([0, 2, 3]),  # "the dog runs"
            np.array([0, 1, 4]),  # "the cat sleeps"
            np.array([0, 2, 4]),  # "the dog sleeps"
        ]
        y_train = np.array([0, 1, 0, 1])  # 0: animal, 1: action
        
        print(f"Training on {len(X_train)} samples...")
        classifier.fit(X_train, y_train, epochs=50)
        
        # Test
        test_sentence = np.array([0, 1, 3])
        prediction = classifier.predict(test_sentence)
        print(f"\nTest: 'the cat runs' -> Class {prediction}")


def demo_qnlp():
    """Run full QNLP demonstration."""
    demo = QNLPDemo()
    demo.demo_sentence_encoding()
    demo.demo_text_classification()


if __name__ == "__main__":
    demo_qnlp()
