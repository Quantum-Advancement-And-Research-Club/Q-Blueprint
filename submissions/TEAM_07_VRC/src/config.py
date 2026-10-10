"""All physical constants and design choices in one place.
Internal units: time in ns, angular frequency in rad/ns. Hz only at I/O."""
from dataclasses import dataclass
import numpy as np

TWO_PI = 2.0 * np.pi

def hz_to_radns(f_hz):
    """Ordinary frequency [Hz] -> angular frequency [rad/ns]."""
    return TWO_PI * np.asarray(f_hz) * 1e-9

def radns_to_hz(w):
    return np.asarray(w) / TWO_PI * 1e9

@dataclass(frozen=True)
class Config:
    # --- specified by the problem statement ---
    alpha_hz: float = -300e6        # anharmonicity alpha/2pi
    t_gate_ns: float = 20.0         # gate duration
    T1_ns: float = 100e3            # 100 us
    T2_ns: float = 80e3             # 80 us
    t_shot_s: float = 500e-6        # repetition time per shot
    p10: float = 0.01               # P(1|0)
    p01: float = 0.03               # P(0|1); |2> reads as 1
    day_s: float = 24 * 3600.0
    eps_th: float = 1e-3
    # drift model
    xf_sigma_hz: float = 150e3
    xf_tau_s: float = 2 * 3600.0
    rtn_amp_hz: float = 0.8e6
    rtn_dwell_s: float = 4 * 3600.0
    gain_sin_amp: float = 0.02
    xg_sigma: float = 0.005
    xg_tau_s: float = 3 * 3600.0
    official_seed: int = 2026
    # --- design choices (documented in the report) ---
    sigma_frac: float = 0.25        # sigma = tg/4 (specified)
    n_steps: int = 200              # propagator steps per gate (dt = 0.1 ns); convergence-tested
    drift_dt_s: float = 1.0         # drift sampling grid
    drift_margin_s: float = 3600.0  # trace extends past 24 h so calibration at end is defined

    @property
    def alpha(self):  # rad/ns
        return hz_to_radns(self.alpha_hz)
    @property
    def sigma_ns(self):
        return self.sigma_frac * self.t_gate_ns

CFG = Config()
