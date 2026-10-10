"""Reproducibility test - bitwise seed test."""
import sys
sys.path.insert(0, "src")

from q_autopilot_backend.sim.policy import simulate_day
from q_autopilot_backend.api.schemas import PolicyId


def test_seed_reproducibility():
    for seed in [2026, 1, 42]:
        r1 = simulate_day(PolicyId.P2, {"check_interval_min": 5, "trigger_threshold": 3}, seed)
        r2 = simulate_day(PolicyId.P2, {"check_interval_min": 5, "trigger_threshold": 3}, seed)
        assert r1.mean_eps == r2.mean_eps, f"Seed {seed}: mean_eps mismatch"
        assert r1.calib_fraction == r2.calib_fraction, f"Seed {seed}: calib_fraction mismatch"
        assert r1.t_h == r2.t_h, f"Seed {seed}: t_h mismatch"
        assert r1.delta_true_khz == r2.delta_true_khz, f"Seed {seed}: delta_true_khz mismatch"
        assert r1.gain_true == r2.gain_true, f"Seed {seed}: gain_true mismatch"
        assert r1.eps_oracle == r2.eps_oracle, f"Seed {seed}: eps_oracle mismatch"
        assert r1.qaoa_ratio == r2.qaoa_ratio, f"Seed {seed}: qaoa_ratio mismatch"
        print(f"Seed {seed}: ✓ bitwise identical")


def test_prng_reproducibility():
    from q_autopilot_backend.sim.prng import mulberry32, normal_random
    for seed in [2026, 1, 42]:
        rng1 = mulberry32(seed)
        rng2 = mulberry32(seed)
        for _ in range(100):
            assert rng1() == rng2(), f"Seed {seed}: mulberry32 mismatch"
        # Test normal_random
        rng1 = mulberry32(seed + 1000)
        rng2 = mulberry32(seed + 1000)
        for _ in range(100):
            assert normal_random(rng1) == normal_random(rng2), f"Seed {seed}: normal_random mismatch"
        print(f"Seed {seed}: PRNG ✓ bitwise identical")


if __name__ == "__main__":
    test_prng_reproducibility()
    test_seed_reproducibility()
    print("\nAll reproducibility tests passed!")