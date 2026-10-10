"""Batched piecewise-constant propagator U = U_N ... U_1, U_k = exp(-i H_k dt)."""
import numpy as np
from config import CFG
from transmon import hamiltonian

def expm_herm(H, dt):
    """exp(-i H dt) for Hermitian H of shape (...,3,3) via eigh (exactly unitary)."""
    w, V = np.linalg.eigh(H)
    ph = np.exp(-1j * w * dt)
    return (V * ph[..., None, :]) @ V.conj().swapaxes(-1, -2)

def propagate(delta, g, ox, oy, dt, alpha=None):
    """Full-gate propagator. delta, g: scalars or (B,) arrays; ox, oy: (N,) pulses.
    Returns U of shape (B,3,3) (or (3,3) for scalar inputs). Later steps act on the left."""
    alpha = CFG.alpha if alpha is None else alpha
    scalar = np.ndim(delta) == 0 and np.ndim(g) == 0
    d = np.atleast_1d(np.asarray(delta, float))
    gg = np.atleast_1d(np.asarray(g, float))
    d, gg = np.broadcast_arrays(d, gg)
    H = hamiltonian(d[:, None], gg[:, None], ox[None, :], oy[None, :], alpha)  # (B,N,3,3)
    Uk = expm_herm(H, dt)
    U = Uk[:, 0]
    for k in range(1, Uk.shape[1]):
        U = Uk[:, k] @ U
    return U[0] if scalar else U

def propagate_states(psi0, delta, g, ox, oy, dt, alpha=None):
    """State trajectory (N+1,3) for a scalar-parameter gate; used for P0,P1,P2(t) plots."""
    alpha = CFG.alpha if alpha is None else alpha
    H = hamiltonian(delta, g, ox, oy, alpha)
    Uk = expm_herm(H, dt)
    psi = [np.asarray(psi0, complex)]
    for k in range(len(ox)):
        psi.append(Uk[k] @ psi[-1])
    return np.array(psi)

def idle_propagator(delta, tau_ns, alpha=None):
    """Free evolution (no drive); diagonal, exact."""
    alpha = CFG.alpha if alpha is None else alpha
    d = np.asarray(delta, float)[..., None]
    E = np.stack([0 * d[..., 0], d[..., 0], 2 * d[..., 0] + alpha], axis=-1)
    return np.exp(-1j * E * tau_ns)[..., None, :] * np.eye(3)
