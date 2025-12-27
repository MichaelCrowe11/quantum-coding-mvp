"""
IBM Quantum Experience Quick Setup
Run this script after getting your API token from IBM

STEP 1: Go to https://quantum.ibm.com/
STEP 2: Sign up (free) or log in
STEP 3: Get your API token from the dashboard
STEP 4: Run this script and paste your token
"""

def setup_ibm_quantum():
    print("=" * 60)
    print("IBM QUANTUM SETUP WIZARD")
    print("=" * 60)
    print()
    print("To get your API token:")
    print("1. Go to: https://quantum.ibm.com/")
    print("2. Sign up or log in (free)")
    print("3. Go to your account settings")
    print("4. Copy your API token")
    print()

    token = input("Paste your IBM Quantum API token here: ").strip()

    if not token or len(token) < 20:
        print("\nError: Invalid token. Please get your token from IBM Quantum.")
        return False

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService

        print("\nSaving your token...")
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=token,
            overwrite=True
        )
        print("Token saved successfully!")

        print("\nVerifying connection...")
        service = QiskitRuntimeService()
        backends = service.backends()

        print("\n" + "=" * 60)
        print("CONNECTED TO IBM QUANTUM!")
        print("=" * 60)
        print(f"\nAvailable quantum computers ({len(backends)} total):")

        # Show top backends
        for backend in backends[:7]:
            try:
                n_qubits = backend.num_qubits
                status = backend.status().status_msg
                print(f"  - {backend.name}: {n_qubits} qubits ({status})")
            except:
                print(f"  - {backend.name}")

        print("\nYou can now run quantum circuits on real hardware!")
        print("\nNext step: Run the demo with real hardware:")
        print("  python circuits/quantum_code_similarity.py")
        print()
        print("Or test hardware connection:")
        print("  python -c \"from qiskit_ibm_runtime import QiskitRuntimeService; s = QiskitRuntimeService(); print('Connected!')\"")

        return True

    except ImportError:
        print("\nError: qiskit-ibm-runtime not installed")
        print("Run: pip install qiskit-ibm-runtime")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure your token is correct and try again.")
        return False


def test_quantum_hardware():
    """Test running a simple circuit on real quantum hardware"""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit import QuantumCircuit
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        print("\n" + "=" * 60)
        print("RUNNING ON REAL QUANTUM HARDWARE")
        print("=" * 60)

        service = QiskitRuntimeService()

        # Get least busy backend
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)
        print(f"\nUsing backend: {backend.name} ({backend.num_qubits} qubits)")

        # Create a simple Bell state circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        print("Circuit: Bell state (entangled qubits)")
        print("  H gate on qubit 0")
        print("  CNOT from qubit 0 to qubit 1")
        print("  Measure both qubits")

        # Transpile for the backend
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        transpiled = pm.run(qc)

        print(f"\nSubmitting job to {backend.name}...")
        print("(This may take 1-5 minutes depending on queue)")

        # Run the circuit
        sampler = SamplerV2(backend)
        job = sampler.run([transpiled], shots=1024)

        print(f"Job ID: {job.job_id()}")
        print("Waiting for results...")

        result = job.result()
        pub_result = result[0]

        # Get the counts
        counts = pub_result.data.meas.get_counts()

        print("\n" + "=" * 60)
        print("QUANTUM RESULTS FROM REAL HARDWARE")
        print("=" * 60)
        print(f"\nMeasurement counts (1024 shots):")
        for state, count in sorted(counts.items()):
            pct = count / 1024 * 100
            bar = "#" * int(pct / 2)
            print(f"  |{state}>: {count:4d} ({pct:5.1f}%) {bar}")

        print("\nExpected: ~50% |00> and ~50% |11>")
        print("(Bell state creates entanglement)")

        # Calculate entanglement quality
        correlated = counts.get('00', 0) + counts.get('11', 0)
        quality = correlated / 1024 * 100
        print(f"\nEntanglement quality: {quality:.1f}%")

        if quality > 90:
            print("Excellent quantum coherence!")
        elif quality > 80:
            print("Good quantum coherence")
        else:
            print("Some noise present (normal for NISQ devices)")

        return True

    except Exception as e:
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_quantum_hardware()
    else:
        if setup_ibm_quantum():
            print("\n" + "-" * 60)
            response = input("Would you like to run a test on real quantum hardware? (y/n): ")
            if response.lower() == 'y':
                test_quantum_hardware()
