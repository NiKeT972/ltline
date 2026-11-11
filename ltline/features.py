"""Relay-style feature extraction from raw 3-phase waveforms.

Pipeline (what a modern numerical relay does, plus fault-specific features):

1. Frequency tracking: estimate the grid frequency from voltage zero crossings and
   resample to exactly 64 samples per cycle, so cycles line up sample for sample.
2. Per-cycle DFT phasors (fundamental + harmonics) for every phase and the
   residual (neutral) current Ia+Ib+Ic.
3. Sequence components: positive (1), negative (2) and zero (0) sequence.
4. Superimposed ("delta") quantities: each feature is compared to a reference
   window 0.3-0.7 s earlier, so a change stands out even when the absolute level
   is buried in normal unbalance (a 0.4 A ground fault inside 50 A of neutral current).
5. Arc-signature features: cycle-to-cycle randomness, inter-harmonic energy, even
   harmonics / DC (arc asymmetry), zero-current shoulders, high-frequency energy.

Each feature row describes a sliding window of W cycles ending at cycle c, and
uses only samples up to c (causal), so the same code runs online.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPC = 64          # samples per cycle after resampling
W = 4             # window length, cycles (80 ms)
REF_START = 35    # reference window: cycles c-REF_START ... c-REF_END
REF_END = 15
FIRST_CYCLE = REF_START + W   # first cycle with a full reference window
RATED = 200.0
V_NOM = 230.0
A = np.exp(2j * np.pi / 3)


# ----------------------------------------------------------------------------- preprocessing
def estimate_frequency(v: np.ndarray, fs: float) -> float:
    """Grid frequency from rising zero crossings of a voltage, with linear interpolation."""
    s = np.sign(v)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]
    frac = v[idx] / (v[idx] - v[idx + 1])
    tz = (idx + frac) / fs
    if len(tz) < 3:
        return 50.0
    return float((len(tz) - 1) / (tz[-1] - tz[0]))


def resample_cycles(signals: np.ndarray, fs: float, f: float) -> np.ndarray:
    """Resample so that one cycle is exactly SPC samples. Returns (ch, ncyc, SPC)."""
    n = signals.shape[1]
    duration = n / fs
    ncyc = int(np.floor(duration * f)) - 1
    t_new = np.arange(ncyc * SPC) / (SPC * f)
    t_old = np.arange(n) / fs
    out = np.stack([np.interp(t_new, t_old, ch) for ch in signals])
    return out.reshape(signals.shape[0], ncyc, SPC)


# ----------------------------------------------------------------------------- per-cycle quantities
def per_cycle(cyc: np.ndarray) -> dict[str, np.ndarray]:
    """cyc: (6, ncyc, SPC) with channels va, vb, vc, ia, ib, ic."""
    v, i = cyc[:3], cyc[3:]
    res = i.sum(axis=0)                                          # residual current (ncyc, SPC)
    X = np.fft.rfft(np.concatenate([v, i, res[None]]), axis=-1) * (np.sqrt(2) / SPC)
    # X[ch, cyc, h] = rms phasor of harmonic h (h=0 is DC, scaled differently but consistently)
    Vp, Ip, R = X[:3, :, 1], X[3:6, :, 1], X[6, :, 1]
    q = {}
    q["Vph"], q["Iph"] = Vp, Ip
    q["V1"] = (Vp[0] + A * Vp[1] + A**2 * Vp[2]) / 3
    q["V2"] = (Vp[0] + A**2 * Vp[1] + A * Vp[2]) / 3
    q["I1"] = (Ip[0] + A * Ip[1] + A**2 * Ip[2]) / 3
    q["I2"] = (Ip[0] + A**2 * Ip[1] + A * Ip[2]) / 3
    q["R1"] = R                                                   # residual fundamental (= 3*I0)
    q["R2"] = X[6, :, 2]
    q["R3"] = X[6, :, 3]
    q["R0"] = X[6, :, 0] / np.sqrt(2)                             # residual DC
    q["Idc"] = np.abs(X[3:6, :, 0]) / np.sqrt(2)
    q["Ih2"] = np.abs(X[3:6, :, 2])
    q["Ih3"] = X[3:6, :, 3]
    # cycle-to-cycle randomness: rms of (this cycle - previous cycle)
    d_res = np.diff(res, axis=0, prepend=res[:1])
    q["res_var"] = np.sqrt((d_res ** 2).mean(axis=1))
    d_i = np.diff(i, axis=1, prepend=i[:, :1])
    q["ph_var"] = np.sqrt((d_i ** 2).mean(axis=2))                # (3, ncyc)
    # zero-current shoulders: share of samples near zero, relative to the cycle's peak
    pk = np.abs(i).max(axis=2, keepdims=True) + 1e-9
    q["shoulder"] = (np.abs(i) < 0.03 * pk).mean(axis=2)          # (3, ncyc)
    # high-frequency content: rms of second difference relative to rms
    rms_i = np.sqrt((i ** 2).mean(axis=2)) + 1e-9
    q["ph_rms"] = rms_i
    q["ph_hf"] = np.sqrt((np.diff(i, n=2, axis=2) ** 2).mean(axis=2)) / rms_i
    rms_v = np.sqrt((v ** 2).mean(axis=2)) + 1e-9
    q["v_hf"] = np.sqrt((np.diff(v, n=2, axis=2) ** 2).mean(axis=2)) / rms_v
    q["res"] = res
    return q


# ----------------------------------------------------------------------------- window features
def _win(x, c):
    """Values of the current window (cycles c-W+1 .. c), last axis = cycles."""
    return x[..., c - W + 1: c + 1]


def _ref(x, c):
    return x[..., c - REF_START: c - REF_END]


def _phasor_delta(z, c):
    """|mean(window phasor) - mean(reference phasor)|: the superimposed component."""
    return np.abs(_win(z, c).mean(axis=-1) - _ref(z, c).mean(axis=-1))


def _interharmonic_ratio(res_window: np.ndarray) -> float:
    """Share of residual-current energy that sits between harmonics (random arcing)."""
    spec = np.abs(np.fft.rfft(res_window.ravel())) ** 2
    harmonic_bins = np.arange(0, len(spec), W)     # W-cycle FFT: harmonics every W bins
    total = spec[1:].sum() + 1e-9
    return float((total - spec[harmonic_bins[1:]].sum()) / total)


FEATURE_NAMES: list[str] = []


def window_features(q: dict, c: int) -> dict[str, float]:
    f = {}
    Iph = np.abs(q["Iph"])                     # (3, ncyc) rms
    Iw, Ir = _win(Iph, c).mean(-1), np.median(_ref(Iph, c), -1) + 1e-6
    f["i_max"] = Iw.max() / RATED
    f["i_min_ratio"] = (Iw / Ir).min()
    f["i_max_ratio"] = (Iw / Ir).max()
    f["i_spread"] = (Iw.max() - Iw.min()) / (Iw.mean() + 1e-6)
    I1w = np.abs(_win(q["I1"], c)).mean()
    I1r = np.median(np.abs(_ref(q["I1"], c))) + 1e-6
    f["i1"] = I1w / RATED
    f["i1_ratio"] = I1w / I1r
    f["i2_i1"] = np.abs(_win(q["I2"], c)).mean() / (I1w + 1e-6)
    f["d_i2"] = _phasor_delta(q["I2"], c) / RATED
    # residual (neutral) quantities, in amps: ground faults live here
    f["res1"] = np.abs(_win(q["R1"], c)).mean()
    f["d_res1"] = _phasor_delta(q["R1"], c)
    f["d_res3"] = _phasor_delta(q["R3"], c)
    f["d_res2"] = _phasor_delta(q["R2"], c)
    f["d_res_dc"] = abs(_win(q["R0"], c).mean().real - _ref(q["R0"], c).mean().real)
    rv_w, rv_r = _win(q["res_var"], c), _ref(q["res_var"], c)
    f["res_var"] = rv_w.mean() - np.median(rv_r)
    f["res_var_max"] = rv_w.max() - np.median(rv_r)
    f["res_var_std"] = rv_w.std()
    f["res_ih"] = _interharmonic_ratio(_win(q["res"].T, c).T)
    f["res_ih_ref"] = f["res_ih"] - _interharmonic_ratio(q["res"][c - REF_START: c - REF_START + W])
    # per-phase arc signatures (worst phase)
    pv_w = _win(q["ph_var"], c).mean(-1)
    pv_r = np.median(_ref(q["ph_var"], c), -1)
    rms_w = _win(q["ph_rms"], c).mean(-1)
    f["ph_var_rel"] = ((pv_w - pv_r) / rms_w).max()
    f["ph_var_abs"] = (pv_w - pv_r).max()
    f["shoulder"] = (_win(q["shoulder"], c).mean(-1) - np.median(_ref(q["shoulder"], c), -1)).max()
    f["ph_hf"] = (_win(q["ph_hf"], c).mean(-1) - np.median(_ref(q["ph_hf"], c), -1)).max()
    f["ih2_rel"] = (_win(q["Ih2"], c).mean(-1) / (rms_w + 1e-6)).max()
    f["idc_rel"] = (_win(q["Idc"], c).max(-1) / (rms_w + 1e-6)).max()
    # the weakest phase (a broken conductor): what is left on it after the load is lost.
    # If the source end lies on the ground, this residue is mostly arc current.
    w = int(np.argmin(Iw / Ir))
    f["wk_ratio"] = (Iw / Ir)[w]
    f["wk_amps"] = Iw[w]
    f["wk_dc"] = _win(q["Idc"][w], c).mean() / (Iw[w] + 0.5)
    f["wk_h2"] = _win(q["Ih2"][w], c).mean() / (Iw[w] + 0.5)
    f["wk_h3"] = np.abs(_win(q["Ih3"][w], c)).mean() / (Iw[w] + 0.5)
    f["wk_var"] = _win(q["ph_var"][w], c).mean() / (Iw[w] + 0.5)
    f["wk_hf"] = _win(q["ph_hf"][w], c).mean()
    f["wk_shoulder"] = _win(q["shoulder"][w], c).mean()
    f["wk_angle"] = np.cos(np.angle(_win(q["Iph"][w], c) / (_win(q["Vph"][w], c) + 1e-9))).mean()
    # voltage
    Vph = np.abs(q["Vph"])
    f["v_min_pu"] = _win(Vph, c).mean(-1).min() / V_NOM
    f["v1_pu"] = np.abs(_win(q["V1"], c)).mean() / V_NOM
    f["v2_v1"] = np.abs(_win(q["V2"], c)).mean() / (np.abs(_win(q["V1"], c)).mean() + 1e-6)
    f["v_hf"] = (_win(q["v_hf"], c).max(-1) - np.median(_ref(q["v_hf"], c), -1)).max()
    f["d_v1_pu"] = (np.abs(_win(q["V1"], c)).mean() - np.median(np.abs(_ref(q["V1"], c)))) / V_NOM
    # power factor of the positive-sequence current and its change (motor starts)
    ang = np.angle(_win(q["I1"], c) / (_win(q["V1"], c) + 1e-9))
    ang_r = np.angle(_ref(q["I1"], c) / (_ref(q["V1"], c) + 1e-9))
    f["pf1"] = np.cos(ang).mean()
    f["d_pf1"] = np.cos(ang).mean() - np.median(np.cos(ang_r))
    # trend within window: decaying (inrush) vs growing (arc build-up / overload ramp)
    i1_series = np.abs(_win(q["I1"], c))
    f["i1_trend"] = (i1_series[-1] - i1_series[0]) / (I1r)
    return f


def event_features(signals: np.ndarray, fs: float, step: int = 1) -> tuple[pd.DataFrame, float]:
    """All window features for one event. Returns (DataFrame indexed by end cycle, frequency)."""
    f0 = estimate_frequency(signals[0], fs)
    cyc = resample_cycles(signals.astype(np.float64), fs, f0)
    q = per_cycle(cyc)
    ncyc = cyc.shape[1]
    rows = [window_features(q, c) for c in range(FIRST_CYCLE, ncyc, step)]
    df = pd.DataFrame(rows, index=pd.Index(range(FIRST_CYCLE, ncyc, step), name="cycle"))
    return df, f0


def cycles_to_seconds(cycle_idx, f0: float) -> np.ndarray:
    """End time of a cycle in seconds."""
    return (np.asarray(cycle_idx) + 1) / f0
