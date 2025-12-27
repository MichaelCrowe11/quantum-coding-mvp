"""
Quantum Coding Assistant REST API
Production-ready API for quantum-enhanced code analysis

Endpoints:
- /similarity - Compare two code snippets
- /bugs - Detect potential bugs
- /optimize - Get optimization suggestions
- /health - API health check
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# Flask imports
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Install Flask: pip install flask flask-cors")

# Import quantum modules (relative imports for package)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from circuits.quantum_code_similarity import QuantumCodeSimilarity
    from models.hybrid_quantum_classical import HybridQuantumCodingModel, QuantumCodeBugDetector
    QUANTUM_MODULES_AVAILABLE = True
except ImportError:
    QUANTUM_MODULES_AVAILABLE = False
    print("Quantum modules not found. Run from project root directory.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    use_quantum_hardware: bool = False
    quantum_backend: str = "ibmq_montreal"
    max_code_length: int = 10000
    cache_enabled: bool = True
    rate_limit: int = 100  # requests per minute


class QuantumCodingAPI:
    """
    Quantum Coding Assistant API

    Provides REST endpoints for quantum-enhanced code analysis
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.app = Flask(__name__)
        CORS(self.app)

        # Initialize quantum components
        self._init_quantum_components()

        # Register routes
        self._register_routes()

        # Request cache
        self.cache = {} if self.config.cache_enabled else None

        # Metrics
        self.metrics = {
            "requests_total": 0,
            "quantum_calls": 0,
            "cache_hits": 0,
            "errors": 0
        }

    def _init_quantum_components(self):
        """Initialize quantum analysis components"""
        if QUANTUM_MODULES_AVAILABLE:
            try:
                self.similarity_analyzer = QuantumCodeSimilarity(
                    n_features=4,
                    use_hardware=self.config.use_quantum_hardware,
                    backend_name=self.config.quantum_backend,
                    shots=1024
                )
                logger.info("Quantum similarity analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quantum similarity: {e}")
                self.similarity_analyzer = None

            try:
                import torch
                self.bug_detector = QuantumCodeBugDetector(n_features=32, n_qubits=4)
                self.hybrid_model = HybridQuantumCodingModel(
                    input_dim=64,
                    quantum_dim=4,
                    task="similarity"
                )
                logger.info("Quantum models initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize quantum models: {e}")
                self.bug_detector = None
                self.hybrid_model = None
        else:
            self.similarity_analyzer = None
            self.bug_detector = None
            self.hybrid_model = None

    def _register_routes(self):
        """Register API routes"""

        @self.app.route("/", methods=["GET"])
        def index():
            return jsonify({
                "name": "Quantum Coding Assistant API",
                "version": "1.0.0",
                "endpoints": [
                    "GET /health - Health check",
                    "POST /similarity - Compare code snippets",
                    "POST /bugs - Detect potential bugs",
                    "POST /optimize - Get optimization suggestions",
                    "GET /metrics - API metrics"
                ],
                "quantum_enabled": self.similarity_analyzer is not None,
                "hardware_mode": self.config.use_quantum_hardware
            })

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "healthy",
                "quantum_available": self.similarity_analyzer is not None,
                "models_loaded": self.bug_detector is not None,
                "timestamp": time.time()
            })

        @self.app.route("/similarity", methods=["POST"])
        def similarity():
            """
            Compare two code snippets for similarity

            Request body:
            {
                "code1": "string",
                "code2": "string",
                "include_circuit": bool (optional)
            }
            """
            self.metrics["requests_total"] += 1

            try:
                data = request.get_json()

                if not data or "code1" not in data or "code2" not in data:
                    return jsonify({
                        "error": "Missing required fields: code1, code2"
                    }), 400

                code1 = data["code1"][:self.config.max_code_length]
                code2 = data["code2"][:self.config.max_code_length]
                include_circuit = data.get("include_circuit", False)

                # Check cache
                cache_key = hash((code1, code2))
                if self.cache and cache_key in self.cache:
                    self.metrics["cache_hits"] += 1
                    return jsonify(self.cache[cache_key])

                # Run quantum analysis
                if self.similarity_analyzer:
                    self.metrics["quantum_calls"] += 1
                    result = self.similarity_analyzer.compute_similarity(
                        code1, code2,
                        return_circuit=include_circuit
                    )
                else:
                    # Fallback to simple comparison
                    result = self._fallback_similarity(code1, code2)

                # Cache result
                if self.cache:
                    self.cache[cache_key] = result

                return jsonify(result)

            except Exception as e:
                self.metrics["errors"] += 1
                logger.error(f"Similarity error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/bugs", methods=["POST"])
        def detect_bugs():
            """
            Detect potential bugs in code

            Request body:
            {
                "code": "string"
            }
            """
            self.metrics["requests_total"] += 1

            try:
                data = request.get_json()

                if not data or "code" not in data:
                    return jsonify({"error": "Missing required field: code"}), 400

                code = data["code"][:self.config.max_code_length]

                if self.bug_detector:
                    import torch
                    self.metrics["quantum_calls"] += 1

                    # Extract features from code
                    features = self._extract_bug_features(code)
                    features_tensor = torch.tensor([features], dtype=torch.float32)

                    result = self.bug_detector(features_tensor)

                    return jsonify({
                        "code_analyzed": True,
                        "quantum_used": True,
                        "analysis": result
                    })
                else:
                    # Fallback analysis
                    result = self._fallback_bug_detection(code)
                    return jsonify({
                        "code_analyzed": True,
                        "quantum_used": False,
                        "analysis": result
                    })

            except Exception as e:
                self.metrics["errors"] += 1
                logger.error(f"Bug detection error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/optimize", methods=["POST"])
        def optimize():
            """
            Get optimization suggestions for code

            Request body:
            {
                "code": "string",
                "optimization_type": "performance" | "memory" | "readability"
            }
            """
            self.metrics["requests_total"] += 1

            try:
                data = request.get_json()

                if not data or "code" not in data:
                    return jsonify({"error": "Missing required field: code"}), 400

                code = data["code"][:self.config.max_code_length]
                opt_type = data.get("optimization_type", "performance")

                # Analyze code for optimization opportunities
                suggestions = self._analyze_for_optimization(code, opt_type)

                return jsonify({
                    "code_analyzed": True,
                    "optimization_type": opt_type,
                    "suggestions": suggestions
                })

            except Exception as e:
                self.metrics["errors"] += 1
                logger.error(f"Optimization error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/batch_similarity", methods=["POST"])
        def batch_similarity():
            """
            Compare query code against multiple candidates

            Request body:
            {
                "query": "string",
                "candidates": ["string", ...],
                "top_k": int (optional, default 5)
            }
            """
            self.metrics["requests_total"] += 1

            try:
                data = request.get_json()

                if not data or "query" not in data or "candidates" not in data:
                    return jsonify({
                        "error": "Missing required fields: query, candidates"
                    }), 400

                query = data["query"][:self.config.max_code_length]
                candidates = [c[:self.config.max_code_length] for c in data["candidates"]]
                top_k = min(data.get("top_k", 5), len(candidates))

                if self.similarity_analyzer:
                    self.metrics["quantum_calls"] += len(candidates)
                    results = self.similarity_analyzer.batch_similarity(
                        query, candidates, top_k
                    )
                else:
                    results = self._fallback_batch_similarity(query, candidates, top_k)

                return jsonify({
                    "query_analyzed": True,
                    "total_candidates": len(candidates),
                    "top_k": top_k,
                    "results": results
                })

            except Exception as e:
                self.metrics["errors"] += 1
                logger.error(f"Batch similarity error: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/metrics", methods=["GET"])
        def metrics():
            """Get API metrics"""
            return jsonify({
                "metrics": self.metrics,
                "cache_size": len(self.cache) if self.cache else 0,
                "uptime": time.time()
            })

    def _extract_bug_features(self, code: str) -> List[float]:
        """Extract features for bug detection (32 features)"""
        features = []

        # Basic metrics
        features.append(len(code) / 1000)  # Normalized length
        features.append(code.count('\n') / 100)  # Line count
        features.append(code.count('if ') / 50)  # Conditionals
        features.append(code.count('for ') / 30)  # Loops
        features.append(code.count('while ') / 20)
        features.append(code.count('try') / 20)  # Error handling
        features.append(code.count('except') / 20)

        # Potential bug indicators
        features.append(code.count('None') / 30)  # Null references
        features.append(code.count('[]') / 20)  # Empty arrays
        features.append(code.count('{}') / 20)  # Empty dicts
        features.append(code.count('==') / 50)  # Comparisons
        features.append(code.count('!=') / 30)
        features.append(code.count('global ') / 10)  # Global state
        features.append(code.count('thread') / 10)  # Threading
        features.append(code.count('lock') / 10)
        features.append(code.count('async') / 20)

        # Memory-related
        features.append(code.count('malloc') / 10)
        features.append(code.count('free') / 10)
        features.append(code.count('new ') / 20)
        features.append(code.count('delete') / 10)
        features.append(code.count('open(') / 20)  # Resource handling
        features.append(code.count('close(') / 20)

        # Complexity indicators
        max_indent = max([len(line) - len(line.lstrip())
                         for line in code.split('\n') if line.strip()], default=0)
        features.append(max_indent / 30)

        # Function complexity
        features.append(code.count('def ') / 30)
        features.append(code.count('return ') / 50)
        features.append(code.count('class ') / 10)

        # String operations (potential injection)
        features.append(code.count('format(') / 20)
        features.append(code.count('% ') / 20)
        features.append(code.count('f"') / 20)

        # Pad to 32 features
        while len(features) < 32:
            features.append(0.0)

        return [min(f, 1.0) for f in features[:32]]

    def _fallback_similarity(self, code1: str, code2: str) -> Dict:
        """Fallback similarity when quantum is unavailable"""
        # Simple Jaccard similarity
        words1 = set(code1.split())
        words2 = set(code2.split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        similarity = intersection / max(union, 1)

        return {
            "similarity": similarity,
            "method": "fallback_jaccard",
            "quantum_used": False,
            "confidence": 0.7
        }

    def _fallback_bug_detection(self, code: str) -> Dict:
        """Fallback bug detection"""
        warnings = []

        if 'None' in code and '== None' in code:
            warnings.append({
                "type": "null_comparison",
                "message": "Use 'is None' instead of '== None'",
                "severity": "low"
            })

        if 'except:' in code or 'except Exception:' in code:
            warnings.append({
                "type": "broad_exception",
                "message": "Avoid catching broad exceptions",
                "severity": "medium"
            })

        if 'global ' in code:
            warnings.append({
                "type": "global_state",
                "message": "Global variables can lead to bugs",
                "severity": "medium"
            })

        return {
            "warnings": warnings,
            "risk_level": "low" if len(warnings) == 0 else "medium",
            "method": "static_analysis"
        }

    def _fallback_batch_similarity(self,
                                    query: str,
                                    candidates: List[str],
                                    top_k: int) -> List[Dict]:
        """Fallback batch similarity"""
        results = []

        for i, candidate in enumerate(candidates):
            sim_result = self._fallback_similarity(query, candidate)
            sim_result["candidate_index"] = i
            results.append(sim_result)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _analyze_for_optimization(self, code: str, opt_type: str) -> List[Dict]:
        """Analyze code for optimization opportunities"""
        suggestions = []

        if opt_type == "performance":
            if 'for ' in code and '.append(' in code:
                suggestions.append({
                    "type": "list_comprehension",
                    "message": "Consider using list comprehension instead of loop + append",
                    "impact": "medium"
                })

            if '+ ' in code and 'str' in code.lower():
                suggestions.append({
                    "type": "string_concatenation",
                    "message": "Use join() for concatenating multiple strings",
                    "impact": "medium"
                })

            if 'in list(' in code or 'in tuple(' in code:
                suggestions.append({
                    "type": "set_lookup",
                    "message": "Use set for O(1) membership testing",
                    "impact": "high"
                })

        elif opt_type == "memory":
            if 'range(' in code and len(code) > 1000:
                suggestions.append({
                    "type": "generator",
                    "message": "Consider using generators for large sequences",
                    "impact": "high"
                })

            if '.read()' in code:
                suggestions.append({
                    "type": "streaming",
                    "message": "Consider streaming large files instead of reading all at once",
                    "impact": "high"
                })

        elif opt_type == "readability":
            if code.count('    ') > 20 or code.count('\t') > 20:
                suggestions.append({
                    "type": "nesting",
                    "message": "High nesting depth - consider extracting functions",
                    "impact": "medium"
                })

            if len([l for l in code.split('\n') if len(l) > 100]) > 5:
                suggestions.append({
                    "type": "line_length",
                    "message": "Multiple long lines - consider breaking up",
                    "impact": "low"
                })

        return suggestions

    def run(self):
        """Run the API server"""
        logger.info(f"Starting Quantum Coding API on {self.config.host}:{self.config.port}")
        logger.info(f"Quantum hardware mode: {self.config.use_quantum_hardware}")

        self.app.run(
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug
        )


def create_app(config: Optional[APIConfig] = None):
    """Factory function for creating the Flask app"""
    api = QuantumCodingAPI(config)
    return api.app


def main():
    """Run the API server"""
    import os

    print("=" * 60)
    print("QUANTUM CODING ASSISTANT API")
    print("=" * 60)

    # Use environment variables for cloud deployment
    port = int(os.environ.get("PORT", 8080))
    use_quantum = os.environ.get("USE_QUANTUM_HARDWARE", "false").lower() == "true"

    config = APIConfig(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("DEBUG", "false").lower() == "true",
        use_quantum_hardware=use_quantum
    )

    api = QuantumCodingAPI(config)

    print("\nAvailable endpoints:")
    print("  GET  /           - API info")
    print("  GET  /health     - Health check")
    print("  POST /similarity - Compare code snippets")
    print("  POST /bugs       - Detect bugs")
    print("  POST /optimize   - Get optimization suggestions")
    print("  GET  /metrics    - API metrics")

    print(f"\nStarting server at http://localhost:{config.port}")
    print("Press Ctrl+C to stop\n")

    api.run()


if __name__ == "__main__":
    main()
