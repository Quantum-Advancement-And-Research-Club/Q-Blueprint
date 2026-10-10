import numpy as np
from .prng import mulberry32, normal_random, ou_step, telegraph_step
from .constants import CONFIG


def generate_drift_trajectory(seed: int, dt_min: float = CONFIG.DT_MIN) -> dict:
    """Generate 24h detuning & gain drift for given seed."""
    rng = mulberry32(seed + 10000)
    n_steps = int(CONFIG.DAY_DURATION_H * 60 / dt_min)
    t_h = np.arange(0, CONFIG.DAY_DURATION_H + dt_min/60, dt_min/60)[:n_steps]
    dt_h = dt_min / 60.0
    
    # Detuning: OU + telegraph
    delta_khz = np.zeros(n_steps)
    telegraph_state = 0.0
    telegraph_states = (0.0, CONFIG.TELEGRAPH_AMPLITUDE_KHZ)
    
    for i in range(1, n_steps):
        z = normal_random(rng)
        delta_khz[i] = ou_step(
            delta_khz[i-1], 0.0, 
            CONFIG.DETUNING_OU_TAU_H, 
            CONFIG.DETUNING_OU_SIGMA_KHZ,
            dt_h, z
        )
        telegraph_state = telegraph_step(
            telegraph_state, telegraph_states,
            CONFIG.TELEGRAPH_RATE_PER_H, dt_h, rng
        )
        delta_khz[i] += telegraph_state
    
    # Gain: sinusoid + OU
    gain = np.zeros(n_steps)
    phase = 2 * np.pi * rng()  # random phase
    for i in range(n_steps):
        t = t_h[i]
        sinusoid = CONFIG.GAIN_SINUSOID_AMP * np.sin(2 * np.pi * t / CONFIG.GAIN_SINUSOID_PERIOD_H + phase)
        if i == 0:
            ou = normal_random(rng) * CONFIG.GAIN_OU_SIGMA
        else:
            z = normal_random(rng)
            ou = ou_step(
                gain[i-1] - sinusoid, 0.0,
                CONFIG.GAIN_OU_TAU_H, CONFIG.GAIN_OU_SIGMA,
                dt_h, z
            )
        gain[i] = 1.0 + sinusoid + ou
    
    return {"t_h": t_h, "delta_khz": delta_khz, "gain": gain}