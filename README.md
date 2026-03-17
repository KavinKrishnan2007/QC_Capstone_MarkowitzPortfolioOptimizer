# QC_Capstone_AntiMarkovPortfolioOptimizer-
**Quantum Portfolio Optimization Using Variational Quantum Algorithms**

Team: Krishnam, Anushri, Kavin

**Overview**
This project presents a hybrid quantum–classical framework for financial portfolio optimization using a VQE-style approach implemented in PennyLane. The goal is to determine optimal asset allocations that balance expected return and portfolio risk using historical market data.
Traditional Markowitz mean–variance optimization becomes computationally expensive as the number of assets grows. By leveraging a variational quantum approach, we compress high-dimensional portfolio spaces into a small number of qubits, demonstrating the feasibility of near-term quantum methods (NISQ era) for real-world financial decision-making.

**Why We Chose This Problem**
Portfolio optimization is computationally challenging as the number of assets increases. Using hybrid quantum-classical methods allows us to:
- Explore scalable approaches for large asset sets.
- Apply VQE-style variational quantum algorithms to practical finance problems.
- Evaluate quantum performance under real-world hardware constraints.

**Background**
**Classical Portfolio Optimization**
Objective: Minimize risk for a target return using the Markowitz mean–variance framework
Minimize 𝜆𝑤TΣ𝑤 − 𝜇T𝑤
Variables:
𝑤:portfolio weights
μ: expected returns vector
Σ: covariance matrix
λ: risk-aversion parameter
Challenge: Computationally expensive for hundreds of assets due to combinatorial complexity.

**Methodology**
**Data Acquisition**
- Top 256 equity tickers dynamically fetched from Wikipedia.
- Historical price data downloaded for 1 year.
- Computed daily returns, mean returns, and covariance matrix.
**Quantum Model & VQE Implementation**
Simulator: 8-qubit PennyLane simulator
Ansatz: Strongly Entangling Layers with 4 variational layers
Gates: Parameterized RX, RY, RZ rotations + CNOT entanglement gates
Workflow:
- Quantum circuit generates probability-based portfolio weights.
- Weights fed into classical risk–return cost function.
- Parameters optimized via Adam optimizer over 61 iterations.
- Iterative hybrid quantum-classical loop minimizes portfolio risk while maximizing returns.

**Rationale for VQE vs QAOA**
- VQE handles continuous portfolio weights naturally.
- QAOA is binary-focused and requires one qubit per asset → impractical for 256 assets.
- VQE compresses allocations into 8 qubits, making it hardware-efficient and NISQ-compatible.

**Objective Function**
C(θ) = 𝜆𝑤TΣ𝑤 − 𝜇T𝑤
where w = QuantumProbabilities(θ) extracted from the parameterized circuit.

**Post-Processing**
- Portfolio weights ranked
- Capital allocation computed
- Portfolio statistics evaluated
- Growth projection simulated

**Results & Visualization**
**Risk–Return Scatter:**
- Shows individual asset risk vs return.
- Quantum-optimized portfolio lies along the efficient frontier. 
**Portfolio Growth Projection:**
300-day simulation illustrates potential capital growth using expected daily returns.
**Hardware Execution Results:**
- Compared simulator vs 27-qubit quantum processor.
- Minor increase in risk due to hardware noise, but return profile remains superior.
- Top allocations (e.g., CIEN: 11.15%, GLW: 10.34%) highlight the model’s ability to prioritize key assets while maintaining diversification.

**Limitations:**
- Classical covariance estimation remains computationally heavy.
- Historical data assumptions may not reflect future behavior.
- Optimization sensitive to hyperparameters.
- Noise effects exist on hardware but manageable.

**Conclusion**
- Demonstrates scalable quantum portfolio optimization using VQE.
- VQE is effective for continuous weight allocation under NISQ constraints.
- Hardware results validate robustness against decoherence.
- Provides a foundation for future quantum financial applications.

**Requirements**
Python ≥ 3.8
PennyLane
NumPy, Pandas, Matplotlib
Qiskit (for hardware backend execution)

**License**
MIT License
