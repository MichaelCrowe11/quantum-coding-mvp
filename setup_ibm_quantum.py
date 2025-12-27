"""
IBM Quantum Setup - Minimum Viable Quantum Coding Assistant
Ready to run on real quantum hardware TODAY
"""

import os
import json
from pathlib import Path

# Step 1: Install dependencies
REQUIREMENTS = """
qiskit>=1.0.0
qiskit-ibm-runtime>=0.20.0
qiskit-machine-learning>=0.7.0
qiskit-aer>=0.14.0
pennylane>=0.34.0
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
flask>=3.0.0
"""

def create_project_structure():
    """Create quantum coding MVP project structure"""
    base_dir = Path("quantum_coding_mvp")

    directories = [
        "circuits",           # Quantum circuit definitions
        "models",             # Hybrid quantum-classical models
        "api",                # REST API for quantum inference
        "utils",              # Utility functions
        "tests",              # Test cases
        "notebooks",          # Jupyter notebooks
        "data",               # Training data
        "configs",            # Configuration files
    ]

    for dir_name in directories:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    return base_dir

def setup_ibm_quantum_credentials():
    """
    Setup IBM Quantum credentials

    GET YOUR FREE TOKEN:
    1. Go to https://quantum-computing.ibm.com/
    2. Sign up (free)
    3. Go to Account -> API token -> Generate
    4. Copy token and paste below
    """

    config = {
        "ibm_quantum": {
            "channel": "ibm_quantum",
            "token": "YOUR_IBM_QUANTUM_TOKEN_HERE",
            "instance": "ibm-q/open/main"  # Free tier
        },
        "available_backends": {
            "simulator": "ibmq_qasm_simulator",
            "small_hw": "ibm_nairobi",      # 7 qubits - fastest queue
            "medium_hw": "ibmq_montreal",    # 27 qubits - good for learning
            "large_hw": "ibm_brisbane"       # 127 qubits - production
        },
        "default_shots": 1024,
        "max_circuit_depth": 50,
        "error_mitigation": True
    }

    config_path = Path("quantum_coding_mvp/configs/ibm_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved to {config_path}")
    print("\n=== NEXT STEPS ===")
    print("1. Go to https://quantum-computing.ibm.com/")
    print("2. Sign up for free account")
    print("3. Copy your API token")
    print("4. Update the token in configs/ibm_config.json")

    return config_path

def verify_installation():
    """Verify quantum libraries are installed correctly"""
    checks = {}

    try:
        import qiskit
        checks["qiskit"] = f"v{qiskit.__version__}"
    except ImportError:
        checks["qiskit"] = "NOT INSTALLED - run: pip install qiskit"

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        checks["qiskit_ibm_runtime"] = "OK"
    except ImportError:
        checks["qiskit_ibm_runtime"] = "NOT INSTALLED - run: pip install qiskit-ibm-runtime"

    try:
        import pennylane as qml
        checks["pennylane"] = f"v{qml.__version__}"
    except ImportError:
        checks["pennylane"] = "NOT INSTALLED - run: pip install pennylane"

    try:
        import torch
        checks["pytorch"] = f"v{torch.__version__}"
    except ImportError:
        checks["pytorch"] = "NOT INSTALLED - run: pip install torch"

    print("\n=== INSTALLATION STATUS ===")
    for lib, status in checks.items():
        symbol = "✓" if "NOT" not in status else "✗"
        print(f"  {symbol} {lib}: {status}")

    return checks

def test_quantum_connection():
    """Test connection to IBM Quantum"""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        # Try to connect
        service = QiskitRuntimeService()
        backends = service.backends()

        print("\n=== CONNECTED TO IBM QUANTUM ===")
        print(f"Available backends: {len(backends)}")

        for backend in backends[:5]:
            print(f"  - {backend.name}: {backend.num_qubits} qubits")

        return True

    except Exception as e:
        print(f"\n=== CONNECTION FAILED ===")
        print(f"Error: {e}")
        print("\nMake sure you've:")
        print("1. Saved your IBM Quantum token")
        print("2. Run: QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM CODING MVP - SETUP WIZARD")
    print("=" * 60)

    # Create project structure
    base_dir = create_project_structure()
    print(f"\n✓ Created project structure at: {base_dir}")

    # Save requirements
    with open(base_dir / "requirements.txt", "w") as f:
        f.write(REQUIREMENTS)
    print(f"✓ Saved requirements.txt")

    # Setup config
    setup_ibm_quantum_credentials()

    # Verify installation
    verify_installation()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE - Ready to build quantum coding assistant!")
    print("=" * 60)
