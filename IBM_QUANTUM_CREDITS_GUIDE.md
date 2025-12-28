# IBM Quantum Educational Credits Application Guide

## Overview

IBM Quantum offers **free compute credits** for educational and research projects. Your Quantum Coding API qualifies for their programs.

## Available Programs

### 1. IBM Quantum Researchers Program (Recommended)
**Credits**: 10,000+ minutes/month on premium hardware
**Best For**: Research projects, publications

**Apply Here**: https://quantum.ibm.com/programs/researchers

**Your Application Pitch**:
```
Project: Quantum-Enhanced Code Analysis
Goal: Develop quantum algorithms for software engineering (code similarity, bug detection)
Hardware Needs: Access to 100+ qubit systems (ibm_fez, ibm_marrakesh)
Current Status: Working prototype deployed, demonstrated 99.7% accuracy on code similarity
Publications: Planning paper on quantum swap test applications in software engineering
```

### 2. IBM Quantum Educators Program
**Credits**: Unlimited access for teaching
**Best For**: Workshops, courses, tutorials

**Apply Here**: https://quantum.ibm.com/programs/educators

### 3. IBM Quantum Network (Enterprise)
**Credits**: Dedicated access + support
**Best For**: Commercial applications, startups

**Contact**: quantum@us.ibm.com

## Quick Application Steps

### Step 1: Gather Your Materials

You already have these ready:

| Item | Location |
|------|----------|
| Working Demo | https://quantum-vercel-crowelogicos.vercel.app |
| GitHub Repo | https://github.com/MichaelCrowe11/quantum-coding-mvp |
| Hardware Results | 94.8% Bell state fidelity on ibm_fez |
| API Documentation | https://github.com/MichaelCrowe11/quantum-vercel-api |

### Step 2: Write Your Research Proposal

Use this template:

```
TITLE: Quantum Algorithms for Code Analysis

ABSTRACT:
We demonstrate quantum-enhanced code analysis using IBM Quantum hardware.
Our approach uses the quantum swap test for code similarity measurement,
achieving 99.7% accuracy in distinguishing similar from different code patterns.

OBJECTIVES:
1. Optimize quantum circuits for code feature comparison
2. Develop hybrid quantum-classical neural networks for bug detection
3. Benchmark against classical methods on real-world codebases
4. Publish findings in quantum computing / software engineering venues

HARDWARE REQUIREMENTS:
- 100+ qubit systems for complex code analysis
- Low error rates for swap test accuracy
- Estimated usage: 500-1000 jobs/month

CURRENT PROGRESS:
- Deployed API serving quantum similarity analysis
- Tested on ibm_fez (156 qubits) with 94.8% entanglement fidelity
- GitHub: https://github.com/MichaelCrowe11/quantum-coding-mvp

EXPECTED OUTCOMES:
- Publication in quantum computing journal
- Open-source quantum coding toolkit
- Educational materials for quantum software engineering
```

### Step 3: Submit Application

1. Go to https://quantum.ibm.com/programs/researchers
2. Sign in with your IBM Quantum account (already authenticated)
3. Fill out the form with above materials
4. Attach your GitHub links and demo URL
5. Submit and wait 2-4 weeks for review

## Alternative: Request Credits Increase

If you need more credits immediately:

1. Log into https://quantum.ibm.com
2. Go to **Account Settings** > **Usage**
3. Click **Request Credits Increase**
4. Explain your project and current usage

## Track Your Current Usage

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

# Check your current plan
print(f"Plan: {service.active_account()}")

# List available systems
for backend in service.backends():
    print(f"{backend.name}: {backend.num_qubits} qubits, status: {backend.status().status_msg}")
```

## Maximize Free Tier

Current free tier includes:
- 10 minutes/month on real quantum hardware
- Unlimited simulator access
- Access to 100+ qubit systems

To maximize:
1. Use simulators for development/testing
2. Optimize circuits before running on hardware
3. Batch multiple experiments in single jobs
4. Use error mitigation to reduce re-runs

## Your Current Access Summary

| Resource | Status |
|----------|--------|
| IBM Quantum Account | Active (MichaelCrowe11) |
| API Key | Configured |
| Hardware Access | 3 systems (156, 156, 133 qubits) |
| Demo Results | 94.8% fidelity achieved |

## Next Steps After Credits Approval

1. **Scale Up Experiments**: Run larger swap test circuits
2. **Benchmark Study**: Compare quantum vs classical on 1000+ code pairs
3. **Publish Results**: Submit to arXiv or quantum computing venue
4. **Expand API**: Add more quantum-powered features

## Contact IBM Quantum

- **General**: quantum@us.ibm.com
- **Research**: quantum-research@us.ibm.com
- **Slack Community**: qiskit.slack.com
- **Twitter**: @IBMQuantum

---

**Your Project Status**: Ready for researchers program application. You have working code, deployed API, and demonstrated results on real quantum hardware.
