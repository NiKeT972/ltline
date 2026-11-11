"""Physics sanity checks: every event must produce the signature a protection
engineer would expect, and look-alikes must stay within normal operating limits."""
import numpy as np
import pytest

from ltline.sim.feeder import ALL_KINDS, CLASSES, FeederSimulator

SIM = FeederSimulator()
SPC = SIM.cfg.spc
SEEDS = range(12)


def cycle_rms(x):
    n = len(x) // SPC
    return np.sqrt((x[: n * SPC].reshape(n, SPC) ** 2).mean(axis=1))


def before_after(ev, x, settle_cycles=25):
    c = int(ev.onset * SIM.cfg.f0)
    r = cycle_rms(x)
    return r[c - 15: c - 3].mean(), r[c + settle_cycles: c + settle_cycles + 15].mean()


def phase_idx(ev):
    return "abc".index(ev.meta["phase"])


def test_shapes_and_labels():
    for kind in ALL_KINDS:
        ev = SIM.simulate(kind, 0)
        assert ev.signals.shape == (6, SIM.cfg.n)
        assert ev.signals.dtype == np.float32
        assert np.isfinite(ev.signals).all()
        assert ev.label in CLASSES


def test_deterministic_with_seed():
    a = SIM.simulate("downed_conductor", 42).signals
    b = SIM.simulate("downed_conductor", 42).signals
    assert np.array_equal(a, b)


@pytest.mark.parametrize("seed", SEEDS)
def test_healthy_voltage_and_current_in_range(seed):
    ev = SIM.simulate("none", seed)
    v = np.array([cycle_rms(ev.signals[k]).mean() for k in range(3)])
    i = np.array([cycle_rms(ev.signals[3 + k]).mean() for k in range(3)])
    assert np.all((v > 200) & (v < 255)), v            # +/-10% around 230 V
    assert np.all((i > 20) & (i < 220)), i


@pytest.mark.parametrize("seed", SEEDS)
def test_short_circuit_is_many_times_rated(seed):
    ev = SIM.simulate("short_circuit", seed)
    peak = max(before_after(ev, ev.signals[3 + k], 5)[1] for k in range(3))
    assert peak > 2 * SIM.cfg.rated_current   # remote faults can be as low as ~2.3x


@pytest.mark.parametrize("seed", SEEDS)
def test_overload_exceeds_rating(seed):
    ev = SIM.simulate("overload", seed)
    after = np.mean([before_after(ev, ev.signals[3 + k], 45)[1] for k in range(3)])
    assert after > 1.05 * SIM.cfg.rated_current


@pytest.mark.parametrize("seed", SEEDS)
def test_open_conductor_drops_phase_current(seed):
    ev = SIM.simulate("open_conductor", seed)
    p = phase_idx(ev)
    pre, post = before_after(ev, ev.signals[3 + p])
    assert post < 0.45 * pre


@pytest.mark.parametrize("seed", SEEDS)
def test_downed_conductor_current_is_too_small_for_overcurrent_protection(seed):
    ev = SIM.simulate("downed_conductor", seed)
    hif = ev.meta["hif_rms_after_settle"]
    assert 0.0 <= hif < 15.0, hif                      # tenths of an amp to a few amps
    # phase current falls (load lost) instead of rising: a breaker never sees it
    p = phase_idx(ev)
    pre, post = before_after(ev, ev.signals[3 + p])
    assert post < pre


def test_hif_current_is_asymmetric_and_intermittent():
    sim = FeederSimulator()
    rng = np.random.default_rng(0)
    t = np.arange(sim.cfg.n) / sim.cfg.fs
    v = np.sqrt(2) * 230 * np.sin(2 * np.pi * 50 * t)
    i, meta = sim._hif_current(rng, t, v, 0.5, 50, 100)
    on = i[t > 1.0]
    assert abs(on.mean()) > 0                          # asymmetry -> DC component
    half = sim.cfg.spc // 2
    halves = on[: len(on) // half * half].reshape(-1, half)
    peaks = np.abs(halves).max(axis=1)
    assert peaks.std() / peaks.mean() > 0.03           # varies half cycle to half cycle


@pytest.mark.parametrize("seed", SEEDS)
def test_series_arc_creates_flat_shoulders(seed):
    ev = SIM.simulate("series_arc", seed)
    p = phase_idx(ev)
    x = ev.signals[3 + p]
    c = int(ev.onset * SIM.cfg.fs)
    pk = np.abs(x).max()
    near_zero_before = (np.abs(x[c - 1600:c - 64]) < 0.02 * pk).mean()
    near_zero_after = (np.abs(x[c + 320:c + 1856]) < 0.02 * pk).mean()
    assert near_zero_after > near_zero_before


@pytest.mark.parametrize("seed", SEEDS)
def test_capacitor_switching_adds_no_neutral_current(seed):
    ev = SIM.simulate("capacitor_switching", seed)
    res = ev.signals[3] + ev.signals[4] + ev.signals[5]
    pre, post = before_after(ev, res, 1)
    assert post < pre + 20


@pytest.mark.parametrize("kind", ["motor_start", "load_step", "single_phase_load",
                                  "nonlinear_load", "capacitor_switching"])
def test_lookalikes_keep_voltage_normal(kind):
    for seed in SEEDS:
        ev = SIM.simulate(kind, seed)
        v = np.array([before_after(ev, ev.signals[k], 30)[1] for k in range(3)])
        assert np.all(v > 195), (kind, seed, v)
