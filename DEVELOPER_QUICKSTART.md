# Quantum Coding API - Developer Quickstart

## Live API Endpoints

**Base URL**: `https://quantum-vercel-crowelogicos.vercel.app`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/similarity` | POST | Compare code similarity |
| `/api/bugs` | POST | Detect bug patterns |

## 30-Second Integration

### Python
```python
import requests

# Compare two code snippets
result = requests.post(
    "https://quantum-vercel-crowelogicos.vercel.app/api/similarity",
    json={"code1": "def add(a,b): return a+b", "code2": "def sum(x,y): return x+y"}
).json()

print(f"Similarity: {result['similarity']*100:.1f}%")  # Output: 98.5%
```

### JavaScript
```javascript
const result = await fetch('https://quantum-vercel-crowelogicos.vercel.app/api/bugs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({code: 'x = None; x.method()'})
}).then(r => r.json());

console.log(`Risk: ${result.highest_risk}`);  // Output: null_pointer
```

### cURL
```bash
# Health check
curl https://quantum-vercel-crowelogicos.vercel.app/api/health

# Code similarity
curl -X POST https://quantum-vercel-crowelogicos.vercel.app/api/similarity \
  -H "Content-Type: application/json" \
  -d '{"code1": "def f(x): return x*2", "code2": "def g(n): return n*2"}'
```

## Use Cases

1. **Plagiarism Detection**: Compare student submissions
2. **Code Review**: Find similar patterns across codebase
3. **Bug Prevention**: Scan PRs for common bug patterns
4. **Refactoring**: Identify duplicate code for consolidation

## Full SDK (Real Quantum Hardware)

For running on actual IBM Quantum computers:

```bash
git clone https://github.com/MichaelCrowe11/quantum-coding-mvp.git
cd quantum-coding-mvp
pip install qiskit qiskit-ibm-runtime pennylane torch
python SETUP_IBM_QUANTUM_NOW.py  # Enter your IBM Quantum API key
```

## Resources

- **Live Demo**: https://quantum-vercel-crowelogicos.vercel.app
- **API Repo**: https://github.com/MichaelCrowe11/quantum-vercel-api
- **Full SDK**: https://github.com/MichaelCrowe11/quantum-coding-mvp
- **IBM Quantum**: https://quantum.ibm.com (free account)
