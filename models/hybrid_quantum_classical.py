"""
Hybrid Quantum-Classical Coding Model
Combines quantum circuits with classical neural networks

Optimized for NISQ devices (IBM Quantum, IonQ, etc.)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# PennyLane for differentiable quantum circuits
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class QuantumLayer(nn.Module):
    """
    Quantum layer for hybrid model
    Uses PennyLane for differentiable quantum circuits
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if not PENNYLANE_AVAILABLE:
            print("Warning: PennyLane not available, using classical approximation")
            self.use_quantum = False
            self.classical_approx = nn.Sequential(
                nn.Linear(n_qubits, n_qubits * 2),
                nn.Tanh(),
                nn.Linear(n_qubits * 2, n_qubits)
            )
            return

        self.use_quantum = True

        # Create quantum device (simulator for training)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Quantum circuit parameters
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

        # Create QNode
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")

    def _quantum_circuit(self, inputs, weights):
        """
        Variational quantum circuit

        Architecture:
        - Angle encoding of inputs
        - Parameterized rotation layers
        - Entangling layers (CNOT ring)
        - Measurement in Z basis
        """
        # Encode classical inputs
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for layer in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.Rot(
                    weights[layer, i, 0],
                    weights[layer, i, 1],
                    weights[layer, i, 2],
                    wires=i
                )

            # Entangling layer (ring of CNOTs)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        # Return expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum layer

        Args:
            x: Input tensor of shape (batch_size, n_qubits)

        Returns:
            Output tensor of shape (batch_size, n_qubits)
        """
        if not self.use_quantum:
            return self.classical_approx(x)

        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # Run quantum circuit for each sample
            result = self.qnode(x[i], self.weights)
            outputs.append(torch.stack(result))

        return torch.stack(outputs)


class HybridQuantumCodingModel(nn.Module):
    """
    Hybrid Quantum-Classical model for coding assistance

    Architecture:
    1. Classical encoder (transforms code features)
    2. Quantum reasoning layer (captures complex patterns)
    3. Classical decoder (produces final output)

    Tasks:
    - Code similarity detection
    - Bug pattern recognition
    - Code optimization suggestions
    """

    def __init__(self,
                 input_dim: int = 64,
                 quantum_dim: int = 8,
                 n_quantum_layers: int = 2,
                 output_dim: int = 32,
                 task: str = "similarity"):
        super().__init__()

        self.input_dim = input_dim
        self.quantum_dim = min(quantum_dim, 10)  # NISQ limit
        self.output_dim = output_dim
        self.task = task

        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, quantum_dim),
            nn.Tanh()  # Normalize to [-1, 1] for quantum encoding
        )

        # Quantum reasoning layer
        self.quantum_layer = QuantumLayer(
            n_qubits=quantum_dim,
            n_layers=n_quantum_layers
        )

        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(quantum_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

        # Task-specific heads
        if task == "similarity":
            self.task_head = nn.Sequential(
                nn.Linear(output_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif task == "bug_detection":
            self.task_head = nn.Sequential(
                nn.Linear(output_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 2),  # Binary: bug or not
                nn.Softmax(dim=-1)
            )
        elif task == "optimization":
            self.task_head = nn.Sequential(
                nn.Linear(output_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8)  # 8 optimization categories
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input through classical + quantum layers"""
        # Classical encoding
        classical_features = self.encoder(x)

        # Scale to [0, pi] for quantum encoding
        quantum_input = (classical_features + 1) * np.pi / 2

        # Quantum processing
        quantum_output = self.quantum_layer(quantum_input)

        # Decode quantum output
        decoded = self.decoder(quantum_output)

        return decoded

    def forward(self,
                x: torch.Tensor,
                x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: First code embedding (batch_size, input_dim)
            x2: Second code embedding for similarity task (optional)

        Returns:
            Task-specific output
        """
        # Encode first input
        encoded1 = self.encode(x)

        if self.task == "similarity" and x2 is not None:
            # Encode second input
            encoded2 = self.encode(x2)

            # Concatenate for comparison
            combined = torch.cat([encoded1, encoded2], dim=-1)
            return self.task_head(combined)

        elif self.task == "bug_detection":
            return self.task_head(encoded1)

        elif self.task == "optimization":
            return self.task_head(encoded1)

        return encoded1

    def quantum_similarity(self,
                           code1_features: torch.Tensor,
                           code2_features: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum-enhanced similarity between code features

        Args:
            code1_features: Features from first code snippet
            code2_features: Features from second code snippet

        Returns:
            Similarity score [0, 1]
        """
        return self.forward(code1_features, code2_features)


class QuantumCodeBugDetector(nn.Module):
    """
    Specialized quantum model for bug detection

    Uses quantum circuits to recognize complex bug patterns
    that may be missed by classical models
    """

    def __init__(self, n_features: int = 32, n_qubits: int = 6):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits * 2),
            nn.Tanh()
        )

        # Two quantum layers for deeper pattern recognition
        self.quantum1 = QuantumLayer(n_qubits=n_qubits, n_layers=2)
        self.quantum2 = QuantumLayer(n_qubits=n_qubits, n_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 5),  # 5 bug categories
            nn.Softmax(dim=-1)
        )

        self.bug_categories = [
            "no_bug",
            "null_pointer",
            "buffer_overflow",
            "race_condition",
            "memory_leak"
        ]

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Detect potential bugs in code features

        Returns:
            Dictionary with bug probabilities
        """
        # Extract features
        features = self.feature_extractor(x)

        # Split for two quantum paths
        f1, f2 = features.chunk(2, dim=-1)

        # Scale to [0, pi]
        q1_input = (f1 + 1) * np.pi / 2
        q2_input = (f2 + 1) * np.pi / 2

        # Quantum processing
        q1_out = self.quantum1(q1_input)
        q2_out = self.quantum2(q2_input)

        # Combine quantum outputs
        combined = torch.cat([q1_out, q2_out], dim=-1)

        # Classify
        probs = self.classifier(combined)

        # Build result dictionary
        results = []
        for i, p in enumerate(probs):
            result = {
                self.bug_categories[j]: float(p[j])
                for j in range(len(self.bug_categories))
            }
            result["predicted_bug"] = self.bug_categories[torch.argmax(p).item()]
            result["confidence"] = float(torch.max(p))
            results.append(result)

        return results if len(results) > 1 else results[0]


class QuantumModelTrainer:
    """
    Training utilities for hybrid quantum-classical models
    """

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.history = {"train_loss": [], "val_loss": [], "accuracy": []}

    def train_step(self,
                   batch: Tuple[torch.Tensor, ...],
                   task: str = "similarity") -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        if task == "similarity":
            x1, x2, labels = batch
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(x1, x2)
            loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())

        elif task == "bug_detection":
            x, labels = batch
            x = x.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(x)
            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self,
                    dataloader,
                    task: str = "similarity") -> float:
        """Train for one epoch"""
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            loss = self.train_step(batch, task)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.history["train_loss"].append(avg_loss)

        return avg_loss

    def evaluate(self,
                 dataloader,
                 task: str = "similarity") -> Dict:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                if task == "similarity":
                    x1, x2, labels = batch
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(x1, x2)
                    loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())

                    predictions = (outputs.squeeze() > 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                elif task == "bug_detection":
                    x, labels = batch
                    x = x.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(x)
                    loss = F.cross_entropy(outputs, labels)

                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

                total_loss += loss.item()

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(len(dataloader), 1)

        self.history["val_loss"].append(avg_loss)
        self.history["accuracy"].append(accuracy)

        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }

    def save_model(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]
        print(f"Model loaded from {path}")


def demo_hybrid_model():
    """Demonstrate hybrid quantum-classical model"""
    print("=" * 60)
    print("HYBRID QUANTUM-CLASSICAL MODEL DEMO")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        return

    # Create model
    model = HybridQuantumCodingModel(
        input_dim=64,
        quantum_dim=4,  # Small for demo
        n_quantum_layers=2,
        output_dim=32,
        task="similarity"
    )

    print(f"\nModel architecture:")
    print(f"  Input dim: 64")
    print(f"  Quantum qubits: 4")
    print(f"  Quantum layers: 2")
    print(f"  Output dim: 32")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy inputs
    batch_size = 4
    x1 = torch.randn(batch_size, 64)
    x2 = torch.randn(batch_size, 64)

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        similarity = model(x1, x2)

    print(f"Similarity scores: {similarity.squeeze().tolist()}")

    # Bug detection demo
    print("\n--- Bug Detection Demo ---")
    bug_detector = QuantumCodeBugDetector(n_features=32, n_qubits=4)

    bug_input = torch.randn(1, 32)
    bug_result = bug_detector(bug_input)

    print(f"Bug detection result:")
    for bug_type, prob in bug_result.items():
        if bug_type not in ["predicted_bug", "confidence"]:
            print(f"  {bug_type}: {prob:.3f}")
    print(f"  Predicted: {bug_result['predicted_bug']} (confidence: {bug_result['confidence']:.3f})")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return model, bug_detector


if __name__ == "__main__":
    demo_hybrid_model()
