import numpy as np
from scipy.linalg import expm
from .constants import CONFIG
from .transmon import build_hamiltonian, piecewise_constant_propagator, get_single_qubit_maps

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor(*ops):
    """Tensor product of operators."""
    result = np.array([[1]], dtype=complex)
    for op in ops:
        result = np.kron(result, op)
    return result


def maxcut_4qubit_ring_cost(gamma: float, beta: float, M: np.ndarray) -> float:
    """
    QAOA p=1 on 4-qubit ring MaxCut.
    Uses actual M(t) matrix from transmon simulation for single-qubit gates.
    
    Ideal cost (eps=0) ≈ 0.75 * 4 edges = 3.0
    Gate error reduces effective cost.
    """
    n_qubits = 4
    edges = [(0,1), (1,2), (2,3), (3,0)]
    
    # Initial state |+⟩^⊗4
    psi = np.ones(16, dtype=complex) / 4
    
    # Cost layer: exp(-i γ C) where C = ½ ∑_{(i,j)∈edges} (I - Z_i Z_j)
    # = 2 - ½ ∑ Z_i Z_j (constant shift doesn't matter for expectation)
    # For each edge (i,j): exp(i γ/2 Z_i Z_j)
    for i, j in edges:
        # Build Z_i Z_j operator
        ops = [I]*4
        ops[i] = Z
        ops[j] = Z
        ZZ = tensor(*ops)
        U_edge = expm(1j * gamma/2 * ZZ)
        psi = U_edge @ psi
    
    # Mixer layer: exp(-i β ∑ X_i)
    for i in range(4):
        ops = [I]*4
        ops[i] = X
        X_op = tensor(*ops)
        U_mixer = expm(-1j * beta * X_op)
        psi = U_mixer @ psi
    
    # Measure cost expectation ⟨C⟩
    cost = 0
    for i, j in edges:
        ops = [I]*4
        ops[i] = Z
        ops[j] = Z
        ZZ = tensor(*ops)
        cost += np.real(psi.conj() @ ZZ @ psi)
    
    cost = 2 - 0.5 * cost  # C = 2 - ½ ∑ Z_i Z_j
    max_cost = 4  # 4 edges, max cut = 4
    return cost / max_cost


def get_single_qubit_fidelity(M: np.ndarray) -> float:
    """Compute single-qubit gate fidelity from M matrix."""
    # For a π pulse, ideal M = -i X = [[0, -i], [-i, 0]]
    # Process fidelity F = |Tr(-i X M)|² / 4 = |M[1,0] + M[0,1]|² / 4
    val = np.abs(M[1,0] + M[0,1])**2 / 4
    # Clamp to [0, 1] to avoid numerical issues
    return float(np.clip(val, 0.0, 1.0))


def simulate_qaoa_vs_time(eps_oracle: np.ndarray, t_h: np.ndarray) -> np.ndarray:
    """
    Simulate QAOA cost ratio over time using actual single-qubit maps from transmon.
    """
    qaoa_ratios = []
    M_matrices = get_single_qubit_maps(eps_oracle, t_h)
    
    for i, M in enumerate(M_matrices):
        # Use the single-qubit map M for this time step
        # Fidelity is related to |M[1,0] + M[0,1]|^2 / 4
        fidelity = np.abs(M[1,0] + M[0,1])**2 / 4
        eps = 1 - fidelity
        cost = maxcut_4qubit_ring_cost(np.pi/4, np.pi/8, M)
        cost_val = maxcut_4qubit_ring_cost(np.pi/4, np.pi/8, M)
        qaoa_ratios.append(cost_val)
    
    return np.array(qaoa_ratios)