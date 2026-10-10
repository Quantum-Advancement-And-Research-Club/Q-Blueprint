import numpy as np
from typing import Callable


def mulberry32(seed: int) -> Callable[[], float]:
    """Exact port of frontend mulberry32 — bitwise identical output."""
    t = seed & 0xFFFFFFFF
    
    def next_random() -> float:
        nonlocal t
        t = (t + 0x6D2B79F5) & 0xFFFFFFFF
        r = (t ^ (t >> 15)) & 0xFFFFFFFF
        r = (r * (1 | r)) & 0xFFFFFFFF
        r = (r ^ (r + ((r ^ (r >> 7)) & 0xFFFFFFFF) * 61)) & 0xFFFFFFFF
        r = (r ^ (r >> 14)) & 0xFFFFFFFF
        return r / 4294967296.0
    
    return next_random


def normal_random(rng: Callable[[], float]) -> float:
    """Box-Muller — matches frontend."""
    u = 0.0
    v = 0.0
    while u == 0.0:
        u = rng()
    while v == 0.0:
        v = rng()
    return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)


def ou_step(x: float, mu: float, tau: float, sigma: float, dt: float, z: float) -> float:
    """Exact OU step — matches frontend."""
    return mu + (x - mu) * np.exp(-dt / tau) + sigma * np.sqrt(1 - np.exp(-2 * dt / tau)) * z


def telegraph_step(current: float, states: tuple, rate: float, dt: float, rng: Callable[[], float]) -> float:
    """Exact telegraph step — matches frontend."""
    if rng() < rate * dt:
        return states[1] if current == states[0] else states[0]
    return current