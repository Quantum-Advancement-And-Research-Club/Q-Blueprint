import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from config import CFG, hz_to_radns
from drift import make_trace, ou_exact, telegraph
import transmon as T
from propagator import propagate, expm_herm, idle_propagator

# ---------- operators / Hamiltonian ----------
def test_operators():
    assert np.allclose(T.N_OP, np.diag([0, 1, 2]))
    assert np.allclose(T.ANH, np.diag([0, 0, 2]))
    assert np.allclose(T.a @ T.ad - T.ad @ T.a, np.diag([1, 1, -2]))   # truncation artefact only in |2>
    # qubit block: a+a^dag -> sigma_x, i(a^dag-a) -> sigma_y
    assert np.allclose(T.XQ[:2, :2], [[0, 1], [1, 0]])
    assert np.allclose(T.YQ[:2, :2], [[0, -1j], [1j, 0]])
    assert np.isclose(T.XQ[1, 2], np.sqrt(2))      # 1<->2 coupling enhanced by sqrt2

def test_hermitian():
    rng = np.random.default_rng(0)
    H = T.hamiltonian(rng.normal(size=50), rng.normal(size=50), rng.normal(size=50), rng.normal(size=50), CFG.alpha)
    assert np.allclose(H, H.conj().swapaxes(-1, -2), atol=1e-14)

def test_static_spectrum():
    d = hz_to_radns(1.234e6)
    H = T.hamiltonian(d, 1.0, 0.0, 0.0, CFG.alpha)
    assert np.allclose(np.diag(H).real, [0, d, 2 * d + CFG.alpha])

# ---------- pulses ----------
def test_gaussian_area():
    for th in [np.pi / 2, np.pi, 0.3]:
        ox, oy, dt = T.gaussian_pulse(th)
        assert np.isclose(ox.sum() * dt, th, rtol=1e-4)     # midpoint rule, 0.1 ns step
        assert np.allclose(oy, 0)
    ox, oy, dt = T.gaussian_pulse(np.pi, n_steps=4000, beta=0.5)
    assert np.isclose(ox.sum() * dt, np.pi, rtol=1e-8)
    # DRAG quadrature equals -beta * numerical derivative
    num = -0.5 * np.gradient(ox, dt)
    assert np.allclose(oy[5:-5], num[5:-5], atol=1e-5)
    assert np.isclose(T.gaussian_pulse(np.pi)[0].max() / (2 * np.pi) * 1e3, 41.8, atol=0.3)  # MHz

# ---------- propagator ----------
def test_unitary():
    ox, oy, dt = T.gaussian_pulse(np.pi, beta=-0.5)
    d = hz_to_radns(np.linspace(-1e6, 1e6, 9)); g = np.linspace(0.9, 1.1, 9)
    U = propagate(d, g, ox, oy, dt)
    assert U.shape == (9, 3, 3)
    assert np.allclose(U @ U.conj().swapaxes(-1, -2), np.eye(3), atol=1e-12)

def test_free_evolution_limit():
    d = hz_to_radns(0.7e6); n = 100; dt = 0.2
    U = propagate(d, 1.0, np.zeros(n), np.zeros(n), dt)
    assert np.allclose(U, idle_propagator(d, n * dt)[...], atol=1e-12)

def test_two_level_limit():
    """alpha -> -inf, Delta=0: qubit block is exp(-i (g*theta/2) sigma_x)."""
    ox, oy, dt = T.gaussian_pulse(np.pi)
    U = propagate(0.0, 1.0, ox, oy, dt, alpha=-1e5)
    sx = np.array([[0, 1], [1, 0]])
    ref = np.array([[0, -1j], [-1j, 0]])           # exp(-i pi/2 sx)
    assert np.allclose(U[:2, :2], ref, atol=2e-3)
    assert abs(U[2, 0]) < 1e-3
    # gain scales rotation angle: g=1.1 -> 1.1*pi
    U2 = propagate(0.0, 1.1, ox, oy, dt, alpha=-1e5)
    th = 1.1 * np.pi / 2
    assert np.allclose(U2[:2, :2], np.cos(th) * np.eye(2) - 1j * np.sin(th) * sx, atol=2e-3)

def test_ordering():
    """Different quadratures do not commute: U(x then y) != U(y then x), and ordering is later-on-left."""
    n = 50; dt = 0.1
    from propagator import expm_herm
    H1 = T.hamiltonian(0, 1, 0.3, 0, CFG.alpha); H2 = T.hamiltonian(0, 1, 0, 0.3, CFG.alpha)
    Ua = expm_herm(H2, dt) @ expm_herm(H1, dt)
    ox = np.array([0.3, 0.0]); oy = np.array([0.0, 0.3])
    U = propagate(0.0, 1.0, ox, oy, dt)
    assert np.allclose(U, Ua, atol=1e-13)
    assert not np.allclose(Ua, expm_herm(H1, dt) @ expm_herm(H2, dt), atol=1e-6)

def test_time_convergence():
    """Midpoint propagator error should fall ~ 1/N^2."""
    d = hz_to_radns(0.5e6)
    def U(n):
        ox, oy, dt = T.gaussian_pulse(np.pi, n_steps=n, beta=-0.53)
        return propagate(d, 1.02, ox, oy, dt)
    ref = U(3200)
    errs = [np.linalg.norm(U(n) - ref) for n in (25, 50, 100, 200)]
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    assert all(3.0 < r < 5.0 for r in ratios), ratios
    assert errs[-1] < 1e-4      # default n_steps=200 is converged

# ---------- drift ----------
def test_ou_statistics():
    rng = np.random.default_rng(1)
    dt, tau, sig = 1.0, 7200.0, 150e3
    x = ou_exact(2_000_000, dt, tau, sig, rng)      # ~ 278 correlation times
    assert abs(x.std() / sig - 1) < 0.05
    assert abs(x.mean()) < 0.1 * sig
    lag = int(tau / dt)
    ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert abs(ac - np.exp(-1)) < 0.03
    # exact discretisation: lag-1 autocorr is e^{-dt/tau} regardless of dt
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(ac1 - np.exp(-dt / tau)) < 1e-4
    # step-size independence of the stationary law
    x2 = ou_exact(200_000, 60.0, tau, sig, np.random.default_rng(2))
    assert abs(x2.std() / sig - 1) < 0.1

def test_telegraph_statistics():
    rng = np.random.default_rng(3)
    T_tot = 4 * 3600.0 * 20000
    t = np.arange(0, T_tot, 60.0)
    J, sw = telegraph(t, 4 * 3600.0, rng, 0.8e6)
    dwell = np.diff(sw)
    assert abs(dwell.mean() / (4 * 3600) - 1) < 0.03
    assert abs(J.mean() / 0.4e6 - 1) < 0.05            # 50% occupancy
    assert set(np.unique(J)) == {0.0, 0.8e6}
    # exponential dwell: std ~ mean
    assert abs(dwell.std() / dwell.mean() - 1) < 0.05

def test_gain_statistics():
    # xg part: std 0.5 %, tau 3 h
    tr = [make_trace(s, duration_s=24 * 3600 * 1) for s in range(200)]
    xg = np.concatenate([t.xg for t in tr])
    assert abs(xg.std() / 0.005 - 1) < 0.08
    # sinusoid has amplitude 0.02 with period 24h
    g = tr[0].gain - 1 - tr[0].xg
    assert np.allclose(g, 0.02 * np.sin(2 * np.pi * tr[0].t_s / CFG.day_s + tr[0].phi))

def test_official_trace_reproducible_and_independent():
    a, b = make_trace(2026), make_trace(2026)
    assert np.array_equal(a.delta_hz, b.delta_hz) and a.phi == b.phi
    assert not np.array_equal(a.delta_hz, make_trace(2027).delta_hz)
    assert len(a.t_s) == int(CFG.day_s + CFG.drift_margin_s) + 1

if __name__ == "__main__":
    import traceback
    fails = 0
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            try: f(); print("PASS", name)
            except Exception: fails += 1; print("FAIL", name); traceback.print_exc()
    sys.exit(fails)
