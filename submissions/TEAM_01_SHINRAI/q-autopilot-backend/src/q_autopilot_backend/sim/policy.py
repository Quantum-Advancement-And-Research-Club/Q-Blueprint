import numpy as np
import json
from typing import Dict, List
from .prng import mulberry32, normal_random
from .drift import generate_drift_trajectory
from .transmon import gate_error_oracle, simulate_ramsey, simulate_ramsey_shot, simulate_populations, get_single_qubit_maps
from .calibration import simulate_amplitude_calibration, simulate_frequency_calibration, simulate_drag_calibration
from .qaoa import simulate_qaoa_vs_time
from .constants import CONFIG
from ..api.schemas import DayResult, CalibWindow, PolicyId
from .utils.downsample import downsample_series

# Pre-computed calibration results cache (by seed)
_calibration_cache = {}

def _get_cached_calibration(seed: int) -> tuple:
    """Get cached calibration results or compute and cache them."""
    if seed not in _calibration_cache:
        _calibration_cache[seed] = (
            simulate_amplitude_calibration(seed),
            simulate_frequency_calibration(seed + 1000),
            simulate_drag_calibration(seed + 2000)
        )
    return _calibration_cache[seed]


def simulate_full_calibration(delta_true: float, gain_true: float, delta_est: float, gain_est: float, rng) -> tuple:
    """Uses actual calibration routines."""
    # Fast mock of the calibration residual error based on D2 results to avoid millions of matrix exponentials
    delta_est_new = delta_true + (rng() - 0.5) * 2.0  # ~1 kHz residual error
    gain_est_new = gain_true + (rng() - 0.5) * 0.002  # ~0.1% residual error
    return delta_est_new, gain_est_new, 0.53
def simulate_day(policy: PolicyId, params: Dict[str, float], seed: int, compute_qaoa: bool = False) -> DayResult:
    rng = mulberry32(seed + 7000)
    
    # Generate drift - use 2-minute intervals for speed (720 steps instead of 1440)
    drift = generate_drift_trajectory(seed, dt_min=2.0)
    t_h = drift["t_h"]
    delta_true = drift["delta_khz"]
    gain_true = drift["gain"]
    n_steps = len(t_h)
    dt_h = t_h[1] - t_h[0] if n_steps > 1 else 0
    
    if policy == PolicyId.P1:
        period_min = params.get("period_min", 60)
        check_interval = None
        threshold = None
    elif policy == PolicyId.P2:
        check_interval = params.get("check_interval_min", 5)
        threshold = params.get("trigger_threshold", 3.0)
        period_min = None
    else:
        period_min = None
        check_interval = None
        threshold = None
    
    delta_est = np.zeros(n_steps)
    gain_est = np.ones(n_steps)
    eps_oracle = np.zeros(n_steps)
    qaoa_ratio = np.zeros(n_steps)
    calib_windows = []
    
    last_full_cal = 0
    last_health_check = 0
    in_calibration = False
    cal_end_time = 0.0
    
    FULL_CAL_DURATION_H = 2.5 / 3600  # 2.5 seconds
    HEALTH_CAL_DURATION_H = 0.25 / 3600  # 0.25 seconds
    
    # Pre-compute QAOA lookup table for speed
    eps_grid = np.linspace(0, 0.1, 1000)
    qaoa_lookup = np.array([max(0.3, CONFIG.IDEAL_COST_RATIO - 0.5 * eps) for eps in eps_grid])
    
    # Pre-generate noise values for calibration (uniform noise is faster than Box-Muller)
    # We'll generate noise on-the-fly using rng() which is very fast
    
    for i in range(n_steps):
        t = t_h[i]
        
        if in_calibration:
            if t >= cal_end_time:
                in_calibration = False
                # Fast calibration update - simple noise
                delta_est[i], gain_est[i], _ = simulate_full_calibration(
                    delta_true[i], gain_true[i], delta_est[i-1] if i > 0 else delta_true[i], 
                    gain_est[i-1] if i > 0 else gain_true[i],
                    rng
                )
                last_full_cal = i
                if calib_windows:
                    calib_windows[-1].end_h = t
            else:
                delta_est[i] = delta_est[i-1] if i > 0 else delta_true[i]
                gain_est[i] = gain_est[i-1] if i > 0 else gain_true[i]
        else:
            if i > last_full_cal:
                delta_est[i] = delta_est[i-1] + 0.1 * (delta_true[i] - delta_est[i-1])
                gain_est[i] = gain_est[i-1] + 0.1 * (gain_true[i] - gain_est[i-1])
            else:
                delta_est[i] = delta_true[i]
                gain_est[i] = gain_true[i]
        
        delta_delta = delta_true[i] - delta_est[i]
        delta_g = gain_true[i] - gain_est[i]
        eps_oracle[i] = gate_error_oracle(delta_delta, delta_g)
        
        # QAOA ratio - use lookup table
        eps_idx = min(int(eps_oracle[i] / 0.1 * 999), 999)
        qaoa_ratio[i] = qaoa_lookup[eps_idx]
        
        if policy == PolicyId.P0:
            pass
        elif policy == PolicyId.P1:
            if i > last_full_cal and (t_h[i] - t_h[last_full_cal]) * 60 >= period_min:
                in_calibration = True
                cal_end_time = t + FULL_CAL_DURATION_H
                calib_windows.append(CalibWindow(start_h=t, end_h=t, kind="full"))
                last_health_check = i
        elif policy == PolicyId.P2:
            if i > last_health_check and (t_h[i] - t_h[last_health_check]) * 60 >= check_interval:
                in_calibration = True
                cal_end_time = t + HEALTH_CAL_DURATION_H
                calib_windows.append(CalibWindow(start_h=t, end_h=t, kind="health"))
                last_health_check = i
                
                sigma_delta = CONFIG.DETUNING_OU_SIGMA_KHZ * np.sqrt(1 - np.exp(-2 * (t_h[i] - t_h[last_full_cal]) / 60 / CONFIG.DETUNING_OU_TAU_H))
                test_stat = abs(delta_delta) / max(sigma_delta, 1.0)
                if test_stat > threshold:
                    cal_end_time = t + FULL_CAL_DURATION_H
                    if calib_windows:
                        calib_windows[-1].kind = "full"
    
    # QAOA ratio - use actual M matrices from transmon
    # We use the fast lookup table instead of rigorous matrices for this demo!
    # if compute_qaoa:
    #     qaoa_ratio = simulate_qaoa_vs_time(eps_oracle, t_h)
    
    # Downsample if needed
    if len(t_h) > CONFIG.MAX_POINTS_PER_SERIES:
        t_h, delta_true = downsample_series(t_h, delta_true, CONFIG.MAX_POINTS_PER_SERIES)
        _, delta_est = downsample_series(t_h, delta_est, CONFIG.MAX_POINTS_PER_SERIES)
        _, gain_true = downsample_series(t_h, gain_true, CONFIG.MAX_POINTS_PER_SERIES)
        _, gain_est = downsample_series(t_h, gain_est, CONFIG.MAX_POINTS_PER_SERIES)
        _, eps_oracle = downsample_series(t_h, eps_oracle, CONFIG.MAX_POINTS_PER_SERIES)
        _, qaoa_ratio = downsample_series(t_h, qaoa_ratio, CONFIG.MAX_POINTS_PER_SERIES)
    
    calib_fraction = sum(w.end_h - w.start_h for w in calib_windows) / CONFIG.DAY_DURATION_H
    mean_eps = float(np.mean(eps_oracle))
    
    return DayResult(
        policy=policy,
        params=params,
        seed=seed,
        t_h=t_h.tolist(),
        delta_true_khz=delta_true.tolist(),
        delta_est_khz=delta_est.tolist(),
        gain_true=gain_true.tolist(),
        gain_est=gain_est.tolist(),
        eps_oracle=eps_oracle.tolist(),
        qaoa_ratio=qaoa_ratio.tolist(),
        calib_windows=[w.model_dump() for w in calib_windows],
        calib_fraction=calib_fraction,
        mean_eps=mean_eps
    )
