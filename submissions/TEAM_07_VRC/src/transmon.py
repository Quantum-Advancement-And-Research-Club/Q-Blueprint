"""Three-level operators, Hamiltonian (Eq. 1) and Gaussian/DRAG pulses."""
import numpy as np
from scipy.special import erf
from config import CFG, Config

D = 3
a = np.diag(np.sqrt(np.arange(1, D)), k=1).astype(complex)   # a|n> = sqrt(n)|n-1>
ad = a.conj().T
N_OP = ad @ a                       # diag(0,1,2)
ANH = ad @ ad @ a @ a               # diag(0,0,2)
XQ = a + ad                         # drive-x operator
YQ = 1j * (ad - a)                  # drive-y operator
P = np.diag([1, 1, 0]).astype(complex)   # projector on qubit subspace

def hamiltonian(delta, g, ox, oy, alpha):
    """H(t) of Eq. (1), broadcast over any leading shape of delta,g,ox,oy [rad/ns].
    Returns array (..., 3, 3)."""
    delta, g, ox, oy = (np.asarray(v, float)[..., None, None] for v in (delta, g, ox, oy))
    return delta * N_OP + (alpha / 2.0) * ANH + (g / 2.0) * (ox * XQ + oy * YQ)

def gaussian_pulse(theta, cfg: Config = CFG, n_steps=None, beta=0.0):
    """Midpoint-sampled Gaussian X-quadrature and DRAG Y-quadrature [rad/ns].
    Omega_x = A exp(-(t-t0)^2/2s^2) on [0,tg], area over the truncated window = theta.
    Omega_y = -beta * dOmega_x/dt  (beta in ns). Returns (ox, oy, dt)."""
    n = cfg.n_steps if n_steps is None else n_steps
    tg, s = cfg.t_gate_ns, cfg.sigma_ns
    dt = tg / n
    t = (np.arange(n) + 0.5) * dt
    t0 = tg / 2
    A = theta / (s * np.sqrt(2 * np.pi) * erf(tg / (2 * np.sqrt(2) * s)))
    ox = A * np.exp(-(t - t0) ** 2 / (2 * s * s))
    dox = -(t - t0) / s ** 2 * ox          # analytic derivative
    oy = -beta * dox
    return ox, oy, dt
