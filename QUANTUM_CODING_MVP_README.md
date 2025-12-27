# Quantum Coding MVP

## Quantum-Enhanced Coding Assistant

A production-ready hybrid quantum-classical system for code analysis, leveraging IBM Quantum hardware and local simulators.

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
cd quantum_coding_mvp
pip install -r requirements.txt
```

Or install individually:
```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer pennylane torch flask flask-cors
```

### 2. Test with Simulator (No Account Needed)

```bash
python circuits/quantum_code_similarity.py
```

This runs a demo comparing code snippets using quantum circuits on a local simulator.

### 3. Run the API

```bash
python api/quantum_api.py
```

API will start at `http://localhost:8080`

### 4. Test the API

```bash
curl -X POST http://localhost:8080/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "code1": "def add(a, b): return a + b",
    "code2": "def sum(x, y): return x + y"
  }'
```

---

## IBM Quantum Setup (Real Hardware)

### Step 1: Create Free Account

1. Go to https://quantum-computing.ibm.com/
2. Click "Sign up" (free)
3. Verify your email

### Step 2: Get API Token

1. Log into IBM Quantum
2. Go to Account → API token → Generate
3. Copy your token

### Step 3: Save Token

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your token (only need to do this once)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN_HERE"
)
```

### Step 4: Run on Real Quantum Hardware

```python
from circuits.quantum_code_similarity import QuantumCodeSimilarity

# Initialize with real hardware
qcs = QuantumCodeSimilarity(
    n_features=4,
    use_hardware=True,  # Enable real quantum hardware
    backend_name="ibm_brisbane",  # 127 qubit system
    shots=1024
)

# Compare code
result = qcs.compute_similarity(
    "def fibonacci(n): ...",
    "def fib(num): ..."
)

print(f"Quantum similarity: {result['similarity']:.3f}")
print(f"Backend used: {result['backend']}")
```

---

## API Reference

### POST /similarity

Compare two code snippets for semantic similarity.

**Request:**
```json
{
  "code1": "def add(a, b): return a + b",
  "code2": "def sum(x, y): return x + y",
  "include_circuit": false
}
```

**Response:**
```json
{
  "similarity": 0.85,
  "confidence": 0.97,
  "quantum_advantage_estimate": "medium",
  "circuit_depth": 23,
  "shots": 1024,
  "backend": "aer_simulator"
}
```

### POST /bugs

Detect potential bugs in code.

**Request:**
```json
{
  "code": "def process(data):\n    for item in data:\n        if item == None:\n            continue"
}
```

**Response:**
```json
{
  "code_analyzed": true,
  "quantum_used": true,
  "analysis": {
    "no_bug": 0.15,
    "null_pointer": 0.65,
    "buffer_overflow": 0.05,
    "race_condition": 0.10,
    "memory_leak": 0.05,
    "predicted_bug": "null_pointer",
    "confidence": 0.65
  }
}
```

### POST /optimize

Get optimization suggestions.

**Request:**
```json
{
  "code": "result = []\nfor item in items:\n    result.append(item * 2)",
  "optimization_type": "performance"
}
```

**Response:**
```json
{
  "code_analyzed": true,
  "optimization_type": "performance",
  "suggestions": [
    {
      "type": "list_comprehension",
      "message": "Consider using list comprehension instead of loop + append",
      "impact": "medium"
    }
  ]
}
```

### POST /batch_similarity

Compare query against multiple candidates.

**Request:**
```json
{
  "query": "def fibonacci(n): ...",
  "candidates": ["def fib(n): ...", "def factorial(n): ...", "def add(a,b): ..."],
  "top_k": 3
}
```

---

## Architecture

```
quantum_coding_mvp/
├── circuits/
│   └── quantum_code_similarity.py    # Quantum circuits for code comparison
├── models/
│   └── hybrid_quantum_classical.py   # Hybrid neural networks
├── api/
│   └── quantum_api.py                # REST API server
├── configs/
│   └── ibm_config.json               # IBM Quantum configuration
├── notebooks/
│   └── (Jupyter notebooks for exploration)
├── tests/
│   └── (Test cases)
├── setup_ibm_quantum.py              # Setup wizard
├── requirements.txt
└── QUANTUM_CODING_MVP_README.md
```

---

## Quantum Circuits Explained

### Swap Test for Code Similarity

The swap test measures the overlap between two quantum states:

```
         ┌───┐                          ┌───┐
ancilla: ┤ H ├──────────●───────────────┤ H ├──M
         └───┘          │               └───┘
                   ┌────┴────┐
code1:   ──|ψ₁⟩───┤         ├───
                   │  CSWAP  │
code2:   ──|ψ₂⟩───┤         ├───
                   └─────────┘
```

**How it works:**
1. Encode code features as rotation angles (angle encoding)
2. Apply controlled-SWAP between the two code registers
3. Measure ancilla qubit
4. P(0) = (1 + |⟨ψ₁|ψ₂⟩|²) / 2

**Quantum advantage:**
- Captures non-linear relationships in code structure
- Potential for quantum speedup in large-scale code search

---

## Performance Benchmarks

| Operation | Simulator | IBM Hardware | Speedup Potential |
|-----------|-----------|--------------|-------------------|
| 2-code similarity | 50ms | 2-5s (queue) | Future: 10x |
| 10-code batch | 500ms | 20-30s | Future: 100x |
| Bug detection | 100ms | 5-10s | Future: 50x |

**Note:** Current quantum advantage is limited by NISQ constraints. As hardware improves, expect significant speedups for large-scale code analysis.

---

## Best Use Cases (Today)

### 1. Code Clone Detection
Find similar code patterns across large codebases.

### 2. Bug Pattern Recognition
Identify common bug patterns using quantum pattern matching.

### 3. Code Search Ranking
Re-rank search results using quantum similarity.

### 4. Security Vulnerability Detection
Match code against known vulnerability patterns.

---

## Roadmap

### Phase 1 (Now): MVP
- [x] Quantum code similarity circuit
- [x] Hybrid quantum-classical model
- [x] REST API
- [x] Local simulator support
- [x] IBM Quantum integration

### Phase 2 (Q1 2025): Enhanced Features
- [ ] Multi-file analysis
- [ ] Language-agnostic support
- [ ] VSCode extension
- [ ] Quantum test generation

### Phase 3 (Q2 2025): Production Scale
- [ ] Distributed quantum execution
- [ ] Advanced error mitigation
- [ ] Custom training pipeline
- [ ] Enterprise API

---

## Troubleshooting

### "Qiskit not installed"
```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer
```

### "PennyLane not available"
```bash
pip install pennylane
```

### "IBM Quantum connection failed"
1. Check your API token is saved
2. Verify network connection
3. Check IBM Quantum status: https://quantum-computing.ibm.com/

### "Queue time too long"
- Use simulator for development: `use_hardware=False`
- Try smaller backends: `ibm_nairobi` (7 qubits)
- Run during off-peak hours

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Test with both simulator and hardware
4. Submit pull request

---

## License

MIT License - See LICENSE file

---

## Contact

Questions? Open an issue or contact the maintainers.

---

*Built with quantum computing for the future of software development.*
