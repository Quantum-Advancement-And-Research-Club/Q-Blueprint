"""q-autopilot simulation package."""

from .constants import CONFIG, load_config
from .prng import mulberry32, normal_random, ou_step, telegraph_step
from .drift import generate_drift_trajectory
from .transmon import simulate_populations, gate_error_oracle, get_single_qubit_maps
from .calibration import (
    simulate_amplitude_calibration,
    simulate_frequency_calibration,
    simulate_drag_calibration,
)
from .adaptive import simulate_adaptive_run, simulate_precision_vs_shots
from .policy import simulate_day
from .qaoa import maxcut_4qubit_ring_cost, simulate_qaoa_vs_time
from .pareto import compute_pareto

__all__ = [
    "CONFIG",
    "load_config",
    "mulberry32",
    "normal_random",
    "ou_step",
    "telegraph_step",
    "generate_drift_trajectory",
    "simulate_populations",
    "gate_error_oracle",
    "get_single_qubit_maps",
    "simulate_amplitude_calibration",
    "simulate_frequency_calibration",
    "simulate_drag_calibration",
    "simulate_adaptive_run",
    "simulate_precision_vs_shots",
    "simulate_day",
    "maxcut_4qubit_ring_cost",
    "simulate_qaoa_vs_time",
    "compute_pareto",
]