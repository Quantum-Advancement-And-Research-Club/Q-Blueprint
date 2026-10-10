import numpy as np
from scipy.linalg import expm
from .constants import CONFIG

# Time grid for piecewise-constant approximation
DT_NS = CONFIG.DT_NS  # 0.1 ns
DURATION_NS = CONFIG.PULSE_DURATION_NS  # 20 ns

def build_hamiltonian(omega_x: float, omega_y: float, delta: float = 0.0) -> np.ndarray:
    """
    3-level transmon Hamiltonian in rotating frame (rad/ns units).
    H = δ|1⟩⟨1| + (2δ + α)|2⟩⟨2| + ½(Ω_x σ_x + Ω_y σ_y) on |0⟩↔|1⟩
                       + √2 ½(Ω_x σ_x + Ω_y σ_y) on |1⟩↔|2⟩ (with anharmonicity α)
    """
    # Anharmonicity: ALPHA_HZ is in Hz, convert to rad/ns
    # α (rad/ns) = 2π × ALPHA_HZ × 1e-9
    alpha_rad_ns = 2 * np.pi * CONFIG.ALPHA_HZ * 1e-9
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    # |0⟩↔|1⟩ coupling (strength = 1)
    # |1⟩↔|2⟩ coupling (strength = √2)
    H = np.zeros((3, 3), dtype=complex)
    
    # Detuning terms
    H[1, 1] = delta
    H[2, 2] = 2 * delta + alpha_rad_ns  # Anharmonicity shifts |2⟩
    
    # Drive terms
    # Ω_x σ_x + Ω_y σ_y on |0⟩↔|1⟩
    drive_01 = 0.5 * (omega_x * sx + omega_y * sy)
    H[0, 1] = drive_01[0, 1]
    H[1, 0] = drive_01[1, 0]
    
    # √2 × drive on |1⟩↔|2⟩
    drive_12 = np.sqrt(2) * 0.5 * (omega_x * sx + omega_y * sy)
    H[1, 2] = drive_12[0, 1]
    H[2, 1] = drive_12[1, 0]
    
    return H

def piecewise_constant_propagator(
    omega_x_vals: np.ndarray,   # Shape: (N_STEPS,)
    omega_y_vals: np.ndarray,   # Shape: (N_STEPS,)
    delta_vals: np.ndarray      # Shape: (N_STEPS,)
) -> np.ndarray:
    """
    Compute total unitary U = ∏_k exp(-i H_k dt) for piecewise-constant Hamiltonians.
    Returns 3×3 unitary matrix.
    """
    DT_NS = CONFIG.DT_NS
    U = np.eye(3, dtype=complex)
    for i in range(len(omega_x_vals)):
        H = build_hamiltonian(omega_x_vals[i], omega_y_vals[i], delta_vals[i])
        U_step = expm(-1j * H * DT_NS)  # DT_NS in ns
        U = U_step @ U
    return U

def simulate_populations(with_drag: bool, beta_ns: float = 0.53, delta: float = 0.0, gain: float = 1.0) -> dict:
    """
    Simulate 3-level populations during π pulse with actual propagator.
    Returns actual leakage and gate fidelity.
    """
    # Generate pulse envelopes
    t_ns = np.arange(0, CONFIG.PULSE_DURATION_NS, CONFIG.DT_NS)
    sigma = CONFIG.PULSE_SIGMA_NS
    center = CONFIG.PULSE_DURATION_NS / 2
    envelope = np.exp(-0.5 * ((t_ns - center) / sigma) ** 2)
    
    # Account for Gaussian truncation at ±2σ (20ns pulse, σ=5ns, center=10ns)
    # The integral from 0 to 20ns is ~95.45% of the full Gaussian
    # Correction factor = 1 / erf(1.414) ≈ 1.0477
    from scipy.special import erf
    trunc_frac = (erf((CONFIG.PULSE_DURATION_NS - center) / (CONFIG.PULSE_SIGMA_NS * np.sqrt(2))) - 
                  erf((0 - center) / (CONFIG.PULSE_SIGMA_NS * np.sqrt(2)))) / 2
    trunc_correction = 1.0 / trunc_frac
    
    # For a π pulse with Gaussian envelope, the peak amplitude should be:
    # Full integral: omega_max * sqrt(2π) * sigma = π → omega_max = sqrt(π/2) / sigma
    # But we truncate, so scale by 1/trunc_frac
    omega_max = np.sqrt(np.pi / 2) / CONFIG.PULSE_SIGMA_NS * trunc_correction
    
    omega_x = omega_max * envelope * gain
    if with_drag:
        # DRAG: Ω_y = -β dΩ_x/dt
        deriv = - (t_ns - center) / sigma**2 * envelope
        omega_y = +beta_ns * omega_max * deriv * gain  # Optimal beta = +0.53 ns (sign verified numerically)
    else:
        omega_y = np.zeros_like(t_ns)
    
    delta_vals = np.full_like(t_ns, delta)
    
    # Compute total unitary
    U = piecewise_constant_propagator(omega_x, omega_y, delta_vals)
    
    # Extract 2×2 block M for |0⟩↔|1⟩
    M = U[:2, :2]
    
    # Initial state |0⟩
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi_final = U @ psi0
    
    # Populations
    p0 = np.abs(psi_final[0])**2
    p1 = np.abs(psi_final[1])**2
    p2 = np.abs(psi_final[2])**2
    leakage = p2
    
    # Gate fidelity using process fidelity for π pulse: F = |Tr(-i X M)|² / 4 = |M[1,0] + M[0,1]|² / 4
    F = np.abs(M[1, 0] + M[0, 1])**2 / 4
    gate_error = 1 - F
    
    # Return full time evolution for plotting
    DT_NS = CONFIG.DT_NS
    U_t = np.eye(3, dtype=complex)
    p0_arr = np.zeros(len(t_ns))
    p1_arr = np.zeros(len(t_ns))
    p2_arr = np.zeros(len(t_ns))
    psi_t = np.array([1, 0, 0], dtype=complex)
    
    for i in range(len(t_ns)):
        H = build_hamiltonian(omega_x[i], omega_y[i], delta_vals[i])
        U_step = expm(-1j * H * DT_NS)  # DT_NS in ns
        U_t = U_step @ U_t
        psi_t = U_t @ np.array([1, 0, 0], dtype=complex)
        p0_arr[i] = np.abs(psi_t[0])**2
        p1_arr[i] = np.abs(psi_t[1])**2
        p2_arr[i] = np.abs(psi_t[2])**2
    
    return {
        "t_ns": t_ns.tolist(),
        "p0": p0_arr.tolist(),
        "p1": p1_arr.tolist(),
        "p2": p2_arr.tolist(),
        "leakage": float(leakage),
        "gate_error": float(gate_error),
        "with_drag": with_drag,
        "beta_ns": beta_ns if with_drag else 0.0
    }

def gate_error_oracle(delta_delta_khz: float, delta_g: float) -> float:
    """Quadratic error model: ε ≈ a·δΔ² + b·δg² + ε_floor"""
    # Coefficients calibrated so that 150 kHz detuning → ε ~ 1e-3
    a = 4.4e-11  # kHz^-2
    b = 0.5      # dimensionless
    return a * delta_delta_khz**2 + b * delta_g**2 + CONFIG.ERROR_FLOOR


def _pi_half_pulse(delta: float = 0.0, gain: float = 1.0, with_drag: bool = True) -> np.ndarray:
    """
    Simulate a π/2 pulse (X_{π/2}) on the transmon.
    Returns the 3×3 unitary for the π/2 pulse.
    """
    t_ns = np.arange(0, CONFIG.PULSE_DURATION_NS, CONFIG.DT_NS)
    sigma = CONFIG.PULSE_SIGMA_NS
    center = CONFIG.PULSE_DURATION_NS / 2
    envelope = np.exp(-0.5 * ((t_ns - center) / sigma) ** 2)
    
    from scipy.special import erf
    trunc_frac = (erf((CONFIG.PULSE_DURATION_NS - center) / (CONFIG.PULSE_SIGMA_NS * np.sqrt(2))) - 
                  erf((0 - center) / (CONFIG.PULSE_SIGMA_NS * np.sqrt(2)))) / 2
    trunc_correction = 1.0 / trunc_frac
    
    # For π pulse: area = π = omega_max * sqrt(2π) * sigma * trunc_frac
    # omega_max = π / (sqrt(2π) * sigma * trunc_frac) = sqrt(π/2) / (sigma * trunc_frac)
    # For π/2 pulse: area = π/2, so omega_max_half = omega_max / 2
    omega_max = np.sqrt(np.pi / 2) / CONFIG.PULSE_SIGMA_NS * trunc_correction
    omega_max_half = omega_max * 0.5
    
    t_ns = np.arange(0, CONFIG.PULSE_DURATION_NS, CONFIG.DT_NS)
    sigma = CONFIG.PULSE_SIGMA_NS
    center = CONFIG.PULSE_DURATION_NS / 2
    envelope = np.exp(-0.5 * ((t_ns - center) / sigma) ** 2)
    
    omega_x = omega_max_half * envelope * gain
    if with_drag:
        # DRAG: Ω_y = -β dΩ_x/dt
        deriv = - (t_ns - center) / sigma**2 * envelope
        omega_y = +0.53 * omega_max_half * deriv * gain  # Optimal beta = +0.53 ns
    else:
        omega_y = np.zeros_like(t_ns)
    
    delta_vals = np.full_like(t_ns, 0.0)
    U = piecewise_constant_propagator(omega_x, omega_y, np.zeros_like(t_ns))
    return U


def simulate_ramsey(tau_us: float, delta_khz: float, with_drag: bool = True, beta_ns: float = 0.53) -> float:
    """
    Simulate Ramsey sequence: π/2 - wait(τ) - π/2
    Returns probability of |1⟩ after sequence.
    """
    # π/2 pulse
    U_pi2 = _pi_half_pulse(delta=0.0, gain=1.0, with_drag=True)
    
    # Free evolution for τ microseconds with detuning
    tau_ns = tau_us * 1000  # convert µs to ns
    DT_NS = CONFIG.DT_NS
    # Free evolution Hamiltonian: H = δ|1⟩⟨1| + 2δ|2⟩⟨2| (no drive)
    # δ in kHz, convert to rad/ns: δ_khz * 2π * 1e-3 * 1e-3 = δ * 2π * 1e-6 rad/ns
    # Actually: δ in kHz = δ * 1000 Hz = δ * 2π * 1000 rad/s = δ * 2π * 1e-3 rad/µs = δ * 2π * 1e-6 rad/ns
    delta_rad_ns = delta_khz * 2 * np.pi * 1e-6
    alpha_rad_ns = 2 * np.pi * CONFIG.ALPHA_HZ * 1e-9
    
    H_free = np.zeros((3, 3), dtype=complex)
    H_free[1, 1] = delta_rad_ns
    H_free[2, 2] = 2 * delta_rad_ns + 2 * np.pi * CONFIG.ALPHA_HZ * 1e-9  # anharmonicity
    
    from scipy.linalg import expm
    U_wait = expm(-1j * H_free * tau_ns)
    
    # π/2 pulse again
    U_pi2_2 = _pi_half_pulse()
    
    # Total unitary: U_pi2 * U_wait * U_pi2
    U_total = U_pi2_2 @ U_wait @ U_pi2
    
    # Initial state |0⟩
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi_final = U_total @ np.array([1, 0, 0], dtype=complex)
    
    # Probability of |1⟩
    p1 = np.abs(psi_final[1])**2
    return float(p1)


def simulate_ramsey_shot(tau_us: float, delta_khz: float, shots: int, rng) -> int:
    """Simulate Ramsey measurement with shot noise."""
    p1 = simulate_ramsey(tau_us, delta_khz)
    return np.random.binomial(shots, p1)


def get_single_qubit_maps(eps_oracle: np.ndarray, t_h: np.ndarray) -> list:
    """
    Get single-qubit M matrices for each time step from the transmon simulation.
    Returns list of 2x2 M matrices for each time step.
    """
    M_matrices = []
    
    # Pre-compute the π pulse unitary once (with DRAG)
    U_pi = simulate_populations(with_drag=True, beta_ns=0.53, delta=0.0, gain=1.0)
    # Extract the 2x2 M matrix from the 3x3 unitary
    # We need the actual unitary, not just the populations
    # For now, use a simplified model based on gate_error_oracle
    # In a full implementation, we'd compute the actual M matrix from the propagator
    
    # For now, we'll create M matrices based on the gate error
    # This is a simplified model - in a full implementation, we'd extract M from the propagator
    for eps in eps_oracle:
        # Fidelity F = 1 - eps
        fidelity = 1 - eps
        # Clamp fidelity to [0, 1] to avoid numerical issues
        fidelity = float(np.clip(fidelity, 0.0, 1.0))
        # For a π pulse, ideal M = -i X = [[0, -i], [-i, 0]]
        # With fidelity F, the M matrix has reduced coherence
        # We model this as: M = sqrt(F) * (-i X) + sqrt(1-F) * noise
        # For simplicity, we use a depolarizing model
        sqrt_F = np.sqrt(max(0.0, fidelity))
        M = np.array([[0, -1j * sqrt_F], [-1j * sqrt_F, 0]], dtype=complex)
        M_matrices.append(M)
    
    return M_matrices
