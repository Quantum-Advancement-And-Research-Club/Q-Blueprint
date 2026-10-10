"""Seeded drift generators: OU (exact), random telegraph, gain."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter
from config import CFG, Config

def ou_exact(n, dt, tau, sigma, rng, mu=0.0):
    """Exact OU sampling x_{k+1}=mu+(x_k-mu)e^{-dt/tau}+sigma*sqrt(1-e^{-2dt/tau})*xi.
    x_0 is drawn from the stationary law N(mu, sigma^2). Returns n samples."""
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1.0 - a * a)
    x0 = rng.normal(0.0, sigma)
    xi = rng.standard_normal(n - 1)
    y, _ = lfilter([b], [1.0, -a], xi, zi=[a * x0])   # y[k] = a*y[k-1] + b*xi[k]
    return mu + np.concatenate(([x0], y))

def telegraph(t, mean_dwell, rng, amp=1.0):
    """Two-state {0, amp} telegraph; exponential dwell times, stationary 50/50 start.
    (For a memoryless process the residual dwell at t=0 is also exponential.)"""
    s0 = int(rng.integers(0, 2))
    n_exp = int(2 * t[-1] / mean_dwell) + 50
    switch = np.cumsum(rng.exponential(mean_dwell, n_exp))
    assert switch[-1] > t[-1]
    k = np.searchsorted(switch, t, side="right")
    return amp * ((s0 + k) % 2).astype(float), switch[switch <= t[-1]]

@dataclass
class DriftTrace:
    t_s: np.ndarray        # grid [s]
    delta_hz: np.ndarray   # Delta/2pi = xf + J  [Hz]
    gain: np.ndarray       # g(t)
    xf_hz: np.ndarray
    J_hz: np.ndarray
    xg: np.ndarray
    phi: float
    switch_times_s: np.ndarray
    seed: int

    def delta_hz_at(self, t_s):
        return np.interp(t_s, self.t_s, self.delta_hz)
    def gain_at(self, t_s):
        return np.interp(t_s, self.t_s, self.gain)

def make_trace(seed=CFG.official_seed, cfg: Config = CFG, duration_s=None) -> DriftTrace:
    """Independent child streams per process => changing one process never shifts another."""
    dur = (cfg.day_s + cfg.drift_margin_s) if duration_s is None else duration_s
    n = int(round(dur / cfg.drift_dt_s)) + 1
    t = np.arange(n) * cfg.drift_dt_s
    ss = np.random.SeedSequence(seed)
    r_xf, r_j, r_xg, r_phi = [np.random.default_rng(s) for s in ss.spawn(4)]
    xf = ou_exact(n, cfg.drift_dt_s, cfg.xf_tau_s, cfg.xf_sigma_hz, r_xf)
    J, sw = telegraph(t, cfg.rtn_dwell_s, r_j, cfg.rtn_amp_hz)
    xg = ou_exact(n, cfg.drift_dt_s, cfg.xg_tau_s, cfg.xg_sigma, r_xg)
    phi = float(r_phi.uniform(0, 2 * np.pi))
    g = 1.0 + cfg.gain_sin_amp * np.sin(2 * np.pi * t / cfg.day_s + phi) + xg
    return DriftTrace(t, xf + J, g, xf, J, xg, phi, sw, seed)
