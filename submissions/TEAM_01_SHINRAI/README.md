# Closed-Loop Auto-Calibration: Making the Qubit Tune Itself
**Q-Blueprint Quantum Hackathon 2026 - Final Submission**

## 📖 Project Overview
This repository contains our complete submission for the Q-Blueprint Quantum Hackathon. We have developed a closed-loop autonomous calibration agent that actively tracks and mitigates parameter drift in a simulated superconducting transmon qubit.

By employing advanced Bayesian adaptive selection, DRAG pulse shaping, and reactive "health-check" policies, our agent successfully maximizes downstream algorithmic performance (QAOA) while strictly bounding hardware downtime.

---

## 📂 Repository Structure

Our submission is organized to guarantee complete code reproducibility:

```text
.
├── Final_Report.md                      # Deliverable 6: Comprehensive physical and architectural analysis
├── README.md                            # You are here!
├── notebooks/                           # Executable Jupyter Notebooks containing the analysis
│   ├── D1_Plant.nbconvert.ipynb         # Deliverable 1: Transmon simulation & stochastic drift generators
│   ├── D2_Calibration.nbconvert.ipynb   # Deliverable 2: Amplitude, Frequency, and DRAG estimators
│   ├── D3_Adaptive.nbconvert.ipynb      # Deliverable 3: Bayesian adaptive vs. fixed grid analysis
│   ├── D4_Policy_Comparison.nbconvert.ipynb # Deliverable 4: 24h Pareto frontier optimization (P0 vs P1 vs P2)
│   ├── D5_Algorithm_Impact.nbconvert.ipynb  # Deliverable 5: QAOA downstream impact modeling
│   └── requirements.txt                 # Pinned dependencies for reproducibility
└── q-autopilot-backend/                 # Core python package powering the simulation
    ├── pyproject.toml
    └── src/
        └── q_autopilot_backend/
            ├── api/
            └── sim/                     # Contains transmon, drift, calibration, qaoa, and policy logic
```

*Note: For the judges' convenience, all notebooks have been pre-executed so that you can immediately view the finalized plots without needing to run the expensive 24-hour matrix exponential simulations.*

---

## 🚀 How to Reproduce

We have strictly separated our core physics engines from our visualization layers. All the heavy lifting is handled by the `q-autopilot-backend` python package.

### 1. Set up the Environment
We recommend using a clean Python virtual environment.
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install the required dependencies
pip install -r notebooks/requirements.txt
```

### 2. Run the Notebooks
Because the notebooks import the backend simulation engine relatively via `sys.path.append('../q-autopilot-backend/src')`, you must ensure that the `q-autopilot-backend` folder remains in the same root directory as the `notebooks` folder.

You can launch Jupyter and run them interactively:
```bash
cd notebooks
jupyter notebook
```
Execute the notebooks in order from `D1` to `D5`.

---

## 📊 Key Highlights

* **Physics-Driven:** We successfully modeled the transmon as a driven three-level system (incorporating the $|2\rangle$ state) to natively capture leakage errors.
* **Realistic Drift:** Emulated diurnal thermal gain fluctuations and TLS-induced Two-Level-System telegraph noise on the frequency detuning.
* **Pareto Optimality:** Proved mathematically in D4 that a reactive, statistics-based health-check policy (P2) strictly dominates a naive fixed-schedule policy (P1) by minimizing calibration overhead while tightly bounding the Oracle gate error.
* **Downstream Validation:** Demonstrated in D5 that our calibration policy prevents the catastrophic degradation of a 4-qubit MaxCut QAOA algorithm over a 24-hour period.

Thank you to the Q-Blueprint organizers for an incredible hackathon experience!
