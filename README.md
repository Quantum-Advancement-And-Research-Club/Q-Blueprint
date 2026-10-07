# Q-Blueprint

## Closed-Loop Auto-Calibration: Make the Qubit Tune Itself

**Q-Blueprint** is a 4-day quantum computing hackathon challenge focused on **pulse-level control, transmon physics, drift tracking, and adaptive experiment design**.

The challenge is to simulate a drifting three-level transmon and build an autonomous calibration system that keeps single-qubit gates within a target error threshold while operating under a strict shot budget.

The final goal is to connect **hardware-level calibration decisions to algorithm-level performance**.

---

## The Challenge

A superconducting qubit is not a "calibrate once and forget" device.

Its transition frequency and effective drive gain can drift over time due to effects such as TLS defects, charge/flux noise, temperature variations, and electronics drift.

In Q-Blueprint, you will build a simulated control stack that can:

- Model a drifting three-level transmon
- Calibrate pulse amplitude, frequency, and DRAG
- Estimate frequency efficiently using Bayesian adaptive experiment design
- Decide when recalibration is actually necessary
- Track calibration performance over a simulated 24-hour day
- Measure how calibration affects the performance of a quantum algorithm

### The central question

> **How little calibration is sufficient to keep a quantum algorithm within tolerance?**

---

## Physics & Simulation

The challenge uses a **three-level transmon model**, allowing leakage from the computational subspace into the \(|2⟩\) state to be explicitly captured.

The simulation includes:

- Frequency drift
- Drive-gain drift
- Anharmonicity
- Gaussian pulses
- DRAG pulse shaping
- Decoherence
- Measurement errors
- Leakage
- Statistical measurement noise

### Baseline parameters

| Parameter | Value |
|---|---:|
| Anharmonicity | α/2π = −300 MHz |
| Gate duration | 20 ns |
| T₁ | 100 μs |
| T₂ | 80 μs |
| Repetition time | 500 μs/shot |
| Levels retained | 3 |
| Readout error | P(1\|0) = 1%, P(0\|1) = 3% |
| Simulated duration | 24 hours |
| Target gate error | εₜₕ = 10⁻³ |

Every calibration shot advances the simulated clock by **500 μs**, creating a fundamental trade-off between calibration cost and computational availability.

---

## Calibration Tasks

You will implement three main calibration primitives.

### 1. Amplitude Calibration

Estimate and correct π-pulse amplitude errors using Rabi-style measurements and error amplification.

### 2. Frequency Calibration

Use Ramsey measurements to estimate the qubit detuning.

The frequency calibration must address:

- Measurement precision
- Aliasing
- Sign ambiguity
- Shot efficiency

### 3. DRAG Calibration

Estimate an appropriate DRAG coefficient β to reduce leakage into the \(|2⟩\) level.

You must compare the experimentally determined optimum with the theoretical scale:

\[
|\beta| \approx \frac{1}{|\alpha|}
\]

while clearly stating your sign convention.

---

## Adaptive Experiment Design

The frequency calibration must use **Bayesian adaptive experiment design**.

Maintain a posterior distribution:

\[
P(\Delta \mid \text{data})
\]

After each batch of measurements, update the posterior and select the next experiment based on an information criterion such as:

- Expected posterior variance reduction, or
- Expected information gain.

You must compare the adaptive strategy against a **fixed dense grid of delays** using the same total shot budget.

Your analysis should demonstrate whether adaptive experimentation provides a meaningful shot-saving advantage.

---

## Drift Model

All teams use the same official stochastic drift model so that results remain comparable.

The simulated 24-hour environment contains:

### Frequency drift

- Ornstein–Uhlenbeck frequency noise
- Stationary standard deviation: 150 kHz
- Correlation time: 2 h
- Random telegraph noise switching between 0 and +0.8 MHz
- Mean dwell time: 4 h

### Gain drift

The drive gain contains:

- A 24-hour sinusoidal component
- Ornstein–Uhlenbeck noise
- 0.5% OU standard deviation
- 3 h correlation time

**Official drift seed: `2026`**

Teams should also evaluate their policies using additional random seeds.

---

## Calibration Policy

Implement and compare at least three calibration policies:

### P0 — No Recalibration

Perform an initial calibration and never recalibrate again.

### P1 — Fixed Schedule

Recalibrate periodically.

Sweep the recalibration period and determine the trade-off between calibration cost and gate performance.

### P2 — Health-Check Policy

Perform a cheap health check periodically and trigger a full recalibration only when a chosen statistic exceeds a justified threshold.

For every policy, record the full-day evolution of the oracle gate error.

---

## Pareto Analysis

For each policy, calculate:

1. **Time-averaged oracle gate error**
2. **Fraction of the day spent calibrating**

Plot the resulting Pareto frontier.

The analysis must include uncertainty estimates using at least **20 random drift seeds**, in addition to the official seed.

---

## Hardware → Algorithm Connection

The final stage connects calibration performance to an actual quantum algorithm.

You will evaluate a:

### p = 1 QAOA for MaxCut on a 4-qubit ring

Single-qubit gates should use the instantaneous 2 × 2 block obtained from the simulated plant, including:

- Rotation-amplitude error
- Detuning error
- Leakage

Two-qubit RZZ gates are treated as ideal.

The final analysis should show how algorithmic performance changes throughout the simulated day under each calibration policy.

---

## Final Performance Question

Your final analysis should answer the following:

> **How does the calibration policy affect the quality of the quantum algorithm, and what is the minimum calibration effort required to maintain acceptable performance?**

Your final figure should communicate a statement of the form:

> **Policy X spends Y% of the day calibrating and delivers a p = 1 QAOA cost within Z% of the ideal value for W% of the day.**

---

# Deliverables

### D1 — Plant Model

A notebook containing:

- Three-level transmon Hamiltonian
- Drift generators
- Gate-error oracle
- Static validation
- Population dynamics
- Leakage analysis

### D2 — Calibration Primitives

Code and convergence plots for:

- Amplitude calibration
- Frequency calibration
- DRAG calibration

Include comparison of:

\[
\beta_{\mathrm{opt}}
\]

with:

\[
\frac{1}{|\alpha|}
\]

and clearly state the sign convention.

### D3 — Adaptive vs Fixed Experiment Design

Include:

- Precision vs total shots
- Posterior evolution
- Selected adaptive delays
- Quantified shot savings
- Explanation of the information-theoretic origin of the improvement

### D4 — Policy Comparison

Include:

- True vs estimated detuning
- True vs estimated gain
- Oracle gate error vs time
- Pareto frontier
- Seed-to-seed error bars

### D5 — Algorithm-Level Impact

Plot:

- QAOA cost vs time of day
- Performance under each calibration policy
- Comparison against the ideal result

### D6 — 400-Word Technical Memo

Title:

> **Which parameter drift hurts the algorithm most, and what would I change in the hardware or the control stack to need less calibration?**

The memo must cite at least **two external sources**.

---

# Four-Day Roadmap

| Day | Phase | Expected Outcome |
|---|---|---|
| **Day 1** | Think | Understand the theory, build the plant model, validate analytic limits |
| **Day 2** | Build | Complete calibration primitives and baseline pipeline |
| **Day 3** | Simulate | Run policy comparisons, Pareto sweeps and algorithm-level simulations |
| **Day 4** | Refine | Fix weaknesses, complete analysis, prepare slides and live demo |

The final submission deadline is the only binding deadline.

---

# Tools & Requirements

The challenge is designed to run on a **laptop CPU**.

### Recommended environment

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- QuTiP
- Qiskit
- Qiskit Aer
- Qiskit Dynamics

No access to real quantum hardware is required.

A full simulation should run within minutes. If simulations become slow, vectorise the propagator computation or cache intermediate results before scaling up.

---

# Submission Format

Each team must submit:

### 1. Reproducible Jupyter Notebook(s)

The notebooks must be:

- Clean
- Commented
- Reproducible
- Based on fixed random seeds
- Accompanied by a pinned dependency list

### 2. Written Report

A PDF containing the requested analysis sections.

The report must be understandable **without running the code**.

### 3. Final Presentation

A slide deck containing **at most 10 slides**.

Teams should be prepared to explain every component of their implementation and results during the final presentation.

---

# Evaluation

The submission is scored out of **100 marks**.

| Criterion | Marks |
|---|---:|
| Conceptual Depth | 20 |
| Implementation Quality | 20 |
| Hardware–Algorithm Integration | 20 |
| Analysis & Critical Thinking | 20 |
| Communication & Demo | 20 |
| **Total** | **100** |

### Conceptual Depth

Correct understanding of the underlying physics and explanation of results through physical mechanisms rather than code behaviour.

### Implementation Quality

Correct, modular, commented and reproducible implementation with appropriate validation and no unexplained constants.

### Hardware–Algorithm Integration

Demonstrate how hardware-level calibration affects measurable algorithm-level performance.

### Analysis & Critical Thinking

Strong uncertainty quantification, comparison with theory, and honest discussion of limitations and failure cases.

### Communication & Demo

Clear reporting, labelled figures, appropriate captions, and a convincing final presentation/live demonstration.

---

# Bonus Challenges

Bonus tasks are outside the 100 marks and are intended to distinguish closely matched teams.

### Option 1 — Kalman Filter

Replace the health check with a Kalman filter that tracks the OU detuning state and triggers recalibration when predicted gate error exceeds the target threshold.

### Option 2 — Virtual-Z Correction

Remove the AC-Stark phase shift introduced by DRAG using a virtual-Z correction and demonstrate the resulting improvement.

---

# Learning Resources

### Transmon Physics & DRAG

- Motzoi et al. (2009), *Simple Pulses for Elimination of Leakage in Weakly Nonlinear Qubits*
- Chen et al. (2016), *Measuring and Suppressing Quantum State Leakage in a Superconducting Qubit*
- Krantz et al. (2019), *A Quantum Engineer's Guide to Superconducting Qubits*
- Pedersen, Møller & Mølmer (2007), *Fidelity of Quantum Operations*

### Automated Calibration & Adaptive Estimation

- Kelly et al. (2018), *Physical Qubit Calibration on a Directed Acyclic Graph*
- Higgins et al. (2007), *Entanglement-Free Heisenberg-Limited Phase Estimation*
- Granade et al. (2012), *Robust Online Hamiltonian Learning*

### Useful Documentation

- Qiskit Dynamics
- Qiskit Experiments
- QuTiP

---

# 📄 Full Problem Statement

The complete official problem statement is available here:

**[Q-Blueprint Problem Statement](problem-statement/Q-Blueprint-Problem-Statement.pdf)**

---

# Originality & Scientific Integrity

All code, analysis and written explanations must be the team's own work.

Any external:

- Libraries
- Code snippets
- Papers
- AI tools
- Other resources

must be appropriately cited.

Teams must be able to explain every part of their submitted work during the final presentation.

A negative or modest result that is correctly analysed is preferable to an overstated claim.

Report distributions and uncertainty estimates rather than presenting only the best run.

---

# 📐 Reporting Standards

All submitted figures should:

- Label axes with units
- Include legends where appropriate
- Include captions
- State the number of seeds and shots behind every error bar
- Clearly specify whether uncertainties represent standard deviation, standard error, or confidence intervals

---

## Good Luck!

Build the simulator.

Calibrate the qubit.

Track the drift.

Spend fewer shots.

Break fewer gates.

And ultimately—

> **Make the qubit tune itself.**

---

**QARC — Quantum Advancement and Research Club**  
**IIT (ISM) Dhanbad**  
**Concetto'26**
