from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional
from enum import Enum


class PolicyId(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class UncertaintyType(str, Enum):
    SD = "SD"
    SE = "SE"
    CI95 = "CI95"


class StaticPulse(BaseModel):
    t_ns: List[float]
    p0: List[float]
    p1: List[float]
    p2: List[float]
    leakage: float
    gate_error: float
    with_drag: bool
    beta_ns: float


class ValidationCheck(BaseModel):
    name: str
    expected: float
    measured: float
    tolerance: float
    passed: bool
    note: Optional[str] = None


class CalibConvergence(BaseModel):
    kind: Literal["amplitude", "frequency", "drag"]
    shots_cum: List[int]
    estimate: List[float]
    sigma: List[float]
    truth: float
    unit: str
    total_shots: int


class AdaptiveRun(BaseModel):
    delta_grid_khz: List[float]
    posteriors: List[List[float]]
    tau_us: List[float]
    shots_cum: List[int]
    truth_khz: float
    estimate_khz: float
    sigma_khz: float


class PrecisionVsShots(BaseModel):
    shots: List[int]
    adaptive_sigma_khz: List[float]
    fixed_sigma_khz: List[float]
    n_repeats: int
    uncertainty: UncertaintyType
    shot_saving_factor: float


class CalibWindow(BaseModel):
    start_h: float
    end_h: float
    kind: Literal["health", "full"]


class DayResult(BaseModel):
    policy: PolicyId
    params: Dict[str, float]
    seed: int
    t_h: List[float]
    delta_true_khz: List[float]
    delta_est_khz: List[float]
    gain_true: List[float]
    gain_est: List[float]
    eps_oracle: List[float]
    qaoa_ratio: List[float]
    calib_windows: List[CalibWindow]
    calib_fraction: float
    mean_eps: float


class ParetoPoint(BaseModel):
    policy: PolicyId
    label: str
    params: Dict[str, float]
    calib_fraction_mean: float
    calib_fraction_err: float
    mean_eps_mean: float
    mean_eps_err: float
    n_seeds: int
    uncertainty: UncertaintyType
    on_frontier: bool


class HardwareAlgoSummary(BaseModel):
    policy: PolicyId
    calib_percent: float
    z_percent: float
    within_z_percent_of_day: float
    ideal_cost_ratio: float


class DriftAblation(BaseModel):
    source: str
    qaoa_loss_percent: float
    err: float
    n_seeds: int


class PolicyInfo(BaseModel):
    id: PolicyId
    name: str
    paramSchema: List[Dict]