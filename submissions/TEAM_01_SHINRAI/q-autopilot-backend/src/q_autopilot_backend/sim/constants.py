import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Transmon
    ALPHA_HZ: float = -300e6
    T1: float = 100e-6
    T2: float = 80e-6
    # Pulse
    PULSE_DURATION_NS: float = 20.0
    PULSE_SIGMA_NS: float = 5.0
    DT_NS: float = 0.1
    # Drift
    DETUNING_OU_SIGMA_KHZ: float = 150.0
    DETUNING_OU_TAU_H: float = 2.0
    TELEGRAPH_AMPLITUDE_KHZ: float = 800.0
    TELEGRAPH_RATE_PER_H: float = 0.25
    GAIN_SINUSOID_AMP: float = 0.02
    GAIN_SINUSOID_PERIOD_H: float = 24.0
    GAIN_OU_SIGMA: float = 0.005
    GAIN_OU_TAU_H: float = 3.0
    # Calibration
    SHOT_DURATION_US: float = 500.0
    ERROR_THRESHOLD: float = 1e-3
    ERROR_FLOOR: float = 1e-4
    # Simulation
    DAY_DURATION_H: float = 24.0
    DT_MIN: float = 1.0
    OFFICIAL_SEED: int = 2026
    RANDOM_SEEDS: list = None
    MAX_POINTS_PER_SERIES: int = 2000
    # QAOA
    N_QUBITS: int = 4
    P_LAYERS: int = 1
    IDEAL_COST_RATIO: float = 0.75

    def __post_init__(self):
        if self.RANDOM_SEEDS is None:
            object.__setattr__(self, 'RANDOM_SEEDS', list(range(1, 21)))


CONFIG = Config()


def load_config():
    """Override from config.yaml if exists."""
    global CONFIG
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        # Apply overrides (simplified)
        if data:
            for key, val in data.get('transmon', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)
            for key, val in data.get('pulse', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)
            for key, val in data.get('drift', {}).get('detuning', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)
            for key, val in data.get('drift', {}).get('gain', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)
            for key, val in data.get('calibration', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)
            for key, val in data.get('simulation', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)
            for key, val in data.get('qaoa', {}).items():
                if hasattr(CONFIG, key.upper()):
                    object.__setattr__(CONFIG, key.upper(), val)