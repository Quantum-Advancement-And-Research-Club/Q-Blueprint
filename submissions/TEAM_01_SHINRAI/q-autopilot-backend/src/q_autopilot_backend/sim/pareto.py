import numpy as np
from typing import List, Optional
from .policy import simulate_day
from .constants import CONFIG
from ..api.schemas import ParetoPoint, PolicyId, UncertaintyType

PARETO_CONFIGS = [
    # P0
    {"policy": PolicyId.P0, "params": {}, "label": "P0: Never"},
    # P1 periods
    * [{"policy": PolicyId.P1, "params": {"period_min": p}, "label": f"P1: {p}min"}
      for p in [10, 30, 60, 120, 180, 240, 300, 360]],
    # P2 configs
    * [{"policy": PolicyId.P2, "params": {"check_interval_min": c, "trigger_threshold": th},
       "label": f"P2: {c}min/{th}σ"}
      for c in [1, 2, 5, 10, 15, 30] for th in [1, 2, 3, 5]],
]

def run_single_config(config: dict, seed: int) -> tuple:
    """Run one config for one seed."""
    result = simulate_day(config["policy"], config["params"], seed)
    return (config["policy"], config["label"], config["params"],
            result.calib_fraction, result.mean_eps)

def compute_pareto(uncertainty: UncertaintyType = UncertaintyType.SD, seeds: Optional[List[int]] = None) -> List[ParetoPoint]:
    """Full Pareto sweep across all seeds."""
    all_results = []

    # Use provided seeds or default to all seeds
    if seeds is None:
        seeds = [CONFIG.OFFICIAL_SEED] + CONFIG.RANDOM_SEEDS

    for config in PARETO_CONFIGS:
        # Run all seeds sequentially (avoiding joblib serialization issues)
        seed_results = [run_single_config(config, seed) for seed in seeds]

        calib_fracs = [r[3] for r in seed_results]
        mean_eps = [r[4] for r in seed_results]

        calib_mean = float(np.mean(calib_fracs))
        calib_err = float(np.std(calib_fracs, ddof=1))
        eps_mean = float(np.mean(mean_eps))
        eps_err = float(np.std(mean_eps, ddof=1))

        # Adjust error based on uncertainty type
        n = len(seed_results)
        if uncertainty == UncertaintyType.SE:
            calib_err /= np.sqrt(n)
            eps_err /= np.sqrt(n)
        elif uncertainty == UncertaintyType.CI95:
            calib_err *= 1.96 / np.sqrt(n)
            eps_err *= 1.96 / np.sqrt(n)

        # Pareto frontier: non-dominated (lower calib_frac AND lower eps)
        on_frontier = False  # Compute after all points collected

        all_results.append({
            "policy": config["policy"],
            "label": config["label"],
            "params": config["params"],
            "calib_fraction_mean": calib_mean,
            "calib_fraction_err": calib_err,
            "mean_eps_mean": eps_mean,
            "mean_eps_err": eps_err,
            "n_seeds": n,
            "uncertainty": uncertainty,
            "on_frontier": on_frontier
        })

    # Compute Pareto frontier
    points = np.array([(r["calib_fraction_mean"], r["mean_eps_mean"]) for r in all_results])
    for i, r in enumerate(all_results):
        # A point is on frontier if no other point has BOTH lower calib_frac AND lower eps
        dominated = any(
            (points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1] and
             (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1]))
            for j in range(len(points))
        )
        r["on_frontier"] = not dominated

    return [ParetoPoint(**r) for r in all_results]
