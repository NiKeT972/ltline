"""Physics-based simulator for a 3-phase low-tension (LT) distribution feeder.

What it models
--------------
A 415 V line-to-line / 230 V line-to-neutral, 50 Hz feeder supplied from a
distribution transformer (e.g. 250 kVA). The measurement point is the LT bus
at the transformer, where a protection unit would sit. Each event is a few
seconds of three phase voltages and three phase currents, sampled at 3.2 kHz
(64 samples per cycle, a typical relay/PMU rate).

Healthy operation
    Unbalanced, slowly varying loads with a lagging power factor and harmonic
    currents from non-linear loads (3rd, 5th, 7th, 9th). The neutral (residual)
    current is therefore never zero: 5-40 A of residual is normal on an LT feeder.
    The source has transformer impedance, so heavy currents pull the bus voltage down.

Faults (label != "Normal")
    Overload          - sustained balanced current above the feeder rating
    ShortCircuit      - bolted/low-resistance L-G, L-L or 3-phase fault, with DC offset
    OpenConductor     - one phase breaks and the broken ends stay in the air
    DownedConductor   - one phase breaks and falls to the ground: a high-impedance
                        fault (HIF). Load current on that phase is lost and a small,
                        erratic arcing current (tenths of an amp to a few amps) flows
                        to earth through asphalt, soil, grass, sand or concrete.
                        Conventional overcurrent protection cannot see this.
    SeriesArc         - a loose or burnt joint arcing in series with the load

Look-alike disturbances (label "Normal": a good protection system must NOT trip)
    motor_start, capacitor_switching, load_step, single_phase_load, nonlinear_load

The HIF model is the widely used two-diode (Emanuel) arc model with
randomised arc voltages and resistances per half cycle, an initial build-up
period, and random extinctions. All parameters are drawn from ranges reported
in the HIF literature and scaled to 230 V LT networks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter

# ----------------------------------------------------------------------------- constants
CLASSES = ["Normal", "Overload", "ShortCircuit", "OpenConductor", "DownedConductor", "SeriesArc"]
FAULT_KINDS = {
    "overload": "Overload",
    "short_circuit": "ShortCircuit",
    "open_conductor": "OpenConductor",
    "downed_conductor": "DownedConductor",
    "series_arc": "SeriesArc",
}
DISTURBANCE_KINDS = ["none", "motor_start", "capacitor_switching", "load_step",
                     "single_phase_load", "nonlinear_load"]
ALL_KINDS = list(FAULT_KINDS) + DISTURBANCE_KINDS
CHANNELS = ["va", "vb", "vc", "ia", "ib", "ic"]
PHASES = "abc"

# Contact-surface resistance ranges (ohms) for a conductor lying on the ground.
# Higher resistance => smaller, harder-to-detect fault current.
SURFACES = {
    "dry_asphalt": (600.0, 2500.0),
    "dry_concrete": (300.0, 1200.0),
    "dry_grass": (150.0, 700.0),
    "dry_soil": (80.0, 400.0),
    "wet_sand": (40.0, 200.0),
    "wet_soil": (20.0, 100.0),
}


@dataclass
class FeederConfig:
    f0: float = 50.0               # nominal frequency, Hz
    fs: float = 3200.0             # sampling rate, Hz (64 samples / cycle)
    duration: float = 3.0          # seconds per event
    v_ln: float = 230.0            # nominal phase voltage, V rms
    rated_current: float = 200.0   # feeder thermal rating, A rms
    r_source: float = 0.010        # transformer + bus resistance, ohm
    x_source: float = 0.030        # transformer + bus reactance at 50 Hz, ohm
    noise_db: tuple = (45.0, 60.0)  # sensor SNR range, dB
    onset_range: tuple = (0.8, 1.6)  # event start, seconds

    @property
    def n(self) -> int:
        return int(round(self.duration * self.fs))

    @property
    def spc(self) -> int:
        """Samples per cycle."""
        return int(round(self.fs / self.f0))


@dataclass
class Event:
    signals: np.ndarray            # (6, n) float32: va, vb, vc, ia, ib, ic
    fs: float
    kind: str                      # simulator kind, e.g. "downed_conductor"
    label: str                     # class, e.g. "DownedConductor" or "Normal"
    onset: float | None            # seconds; None for kind == "none"
    meta: dict = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.signals.shape[1]) / self.fs

    def channel(self, name: str) -> np.ndarray:
        return self.signals[CHANNELS.index(name)]


# ----------------------------------------------------------------------------- helpers
def _smooth_walk(rng, n, fs, sigma, corner_hz=0.5):
    """Slow random variation (1 + small drift), low-pass filtered white noise."""
    alpha = 1.0 - np.exp(-2 * np.pi * corner_hz / fs)
    w = rng.normal(0.0, 1.0, n)
    out = lfilter([alpha], [1.0, alpha - 1.0], w)   # first-order IIR low-pass
    out /= (out.std() + 1e-12)
    return 1.0 + sigma * out


def _step(t, t0, rise=0.002):
    """Smooth 0 -> 1 step at t0 with a short rise time (switching is not instant)."""
    return 0.5 * (1.0 + np.tanh((t - t0) / (rise / 3.0)))


PHASE_SHIFT = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])


# ----------------------------------------------------------------------------- simulator
class FeederSimulator:
    def __init__(self, cfg: FeederConfig | None = None):
        self.cfg = cfg or FeederConfig()

    # ---- public -------------------------------------------------------------
    def simulate(self, kind: str, rng: np.random.Generator | int | None = None) -> Event:
        if kind not in ALL_KINDS:
            raise ValueError(f"unknown kind {kind!r}; choose from {ALL_KINDS}")
        rng = np.random.default_rng(rng)
        cfg = self.cfg
        t = np.arange(cfg.n) / cfg.fs
        f = cfg.f0 + rng.uniform(-0.15, 0.15)           # grid frequency wanders
        wt = 2 * np.pi * f * t
        onset = None if kind == "none" else float(rng.uniform(*cfg.onset_range))
        meta: dict = {"frequency": f}

        vs = self._source_voltage(rng, t, wt)            # (3, n) EMF behind source impedance
        i_load, load_meta = self._healthy_load(rng, t, wt)
        meta.update(load_meta)
        i_extra = np.zeros_like(i_load)                  # currents added by events
        v_extra = np.zeros_like(vs)                      # voltage transients added at the bus

        # Normal events also get an occasional unrelated small disturbance
        if kind == "none" and rng.random() < 0.3:
            meta["background"] = "small_load_step"
            i_load *= 1 + rng.uniform(-0.08, 0.08) * _step(t, rng.uniform(0.3, cfg.duration - 0.3))[None]

        handler = getattr(self, f"_ev_{kind}")
        handler(rng, t, wt, onset, vs, i_load, i_extra, v_extra, meta)

        i_total = i_load + i_extra
        v_bus = self._bus_voltage(vs, i_total) + v_extra
        sig = np.vstack([v_bus, i_total])
        sig = self._add_sensor_noise(rng, sig)
        label = FAULT_KINDS.get(kind, "Normal")
        return Event(sig.astype(np.float32), cfg.fs, kind, label, onset, meta)

    # ---- healthy network ------------------------------------------------------
    def _source_voltage(self, rng, t, wt):
        cfg = self.cfg
        base = rng.uniform(0.96, 1.05)                   # tap position / grid level
        unbal = 1 + rng.uniform(-0.015, 0.015, 3)        # slight voltage unbalance
        drift = _smooth_walk(rng, len(t), cfg.fs, 0.004, 0.3)
        amp = np.sqrt(2) * cfg.v_ln * base * unbal[:, None] * drift[None]
        v = amp * np.sin(wt[None] + PHASE_SHIFT[:, None])
        # background voltage distortion from other feeders (5th and 7th)
        v += np.sqrt(2) * cfg.v_ln * rng.uniform(0.005, 0.025) * np.sin(5 * (wt[None] + PHASE_SHIFT[:, None]) + rng.uniform(0, 2 * np.pi))
        v += np.sqrt(2) * cfg.v_ln * rng.uniform(0.003, 0.015) * np.sin(7 * (wt[None] + PHASE_SHIFT[:, None]) + rng.uniform(0, 2 * np.pi))
        return v

    def _healthy_load(self, rng, t, wt):
        cfg = self.cfg
        mean_load = rng.uniform(0.25, 0.75) * cfg.rated_current
        unbal = 1 + rng.uniform(-0.25, 0.25, 3)
        i_rms = mean_load * unbal                        # per-phase rms
        pf_angle = np.deg2rad(rng.uniform(12, 35, 3))    # lagging
        harm = {  # fraction of fundamental, per phase
            3: rng.uniform(0.04, 0.20, 3),
            5: rng.uniform(0.03, 0.12, 3),
            7: rng.uniform(0.01, 0.06, 3),
            9: rng.uniform(0.00, 0.03, 3),
        }
        harm_phase = {h: rng.uniform(0, 2 * np.pi, 3) for h in harm}
        i = np.zeros((3, len(t)))
        for k in range(3):
            env = _smooth_walk(rng, len(t), cfg.fs, 0.025, 0.4)
            theta = wt + PHASE_SHIFT[k]
            wave = np.sin(theta - pf_angle[k])
            for h, frac in harm.items():
                wave += frac[k] * np.sin(h * theta - harm_phase[h][k])
            i[k] = np.sqrt(2) * i_rms[k] * env * wave
        meta = {"load_rms": i_rms.round(1).tolist(),
                "pf": np.cos(pf_angle).round(3).tolist(),
                "h3": harm[3].round(3).tolist()}
        return i, meta

    def _bus_voltage(self, vs, i):
        cfg = self.cfg
        ls = cfg.x_source / (2 * np.pi * cfg.f0)
        di = np.gradient(i, 1.0 / cfg.fs, axis=1)
        return vs - cfg.r_source * i - ls * di

    def _add_sensor_noise(self, rng, sig):
        snr_db = rng.uniform(*self.cfg.noise_db)
        rms = np.sqrt((sig ** 2).mean(axis=1, keepdims=True))
        noise = rng.normal(0, 1, sig.shape) * rms * 10 ** (-snr_db / 20)
        return sig + noise

    # ---- faults -----------------------------------------------------------------
    def _ev_none(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        meta["description"] = "steady operation"

    def _ev_overload(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        cfg = self.cfg
        cur = np.sqrt((i_load ** 2).mean(axis=1)).mean()
        target = rng.uniform(1.15, 1.7) * cfg.rated_current
        ramp = rng.uniform(0.15, 0.8)                    # loads pile on over a fraction of a second
        m = target / cur
        env = 1 + (m - 1) * np.clip((t - t0) / ramp, 0, 1)
        i_load *= env[None]
        meta.update(target_rms=round(target, 1), ramp_s=round(ramp, 3))

    def _ev_short_circuit(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        cfg = self.cfg
        ftype = rng.choice(["LG", "LL", "LLL"], p=[0.6, 0.25, 0.15])
        r_line = rng.uniform(0.04, 0.35)                 # cable resistance up to the fault (loop)
        x_line = r_line * rng.uniform(0.3, 0.8)
        r_fault = rng.uniform(0.0, 0.15)                 # low-resistance contact
        r = cfg.r_source + r_line + r_fault
        x = cfg.x_source + x_line
        z = np.hypot(r, x)
        phi = np.arctan2(x, r)
        tau = (x / (2 * np.pi * cfg.f0)) / r             # DC-offset time constant
        on = (t >= t0)
        f = meta["frequency"]
        p = int(rng.integers(3))

        def fault_wave(v_amp, ang):
            # steady AC + decaying DC offset so the current is continuous at t0
            ac = np.sin(2 * np.pi * f * t + ang - phi)
            ac0 = np.sin(2 * np.pi * f * t0 + ang - phi)
            dc = -ac0 * np.exp(-np.clip(t - t0, 0, None) / tau)
            return np.sqrt(2) * v_amp / z * (ac + dc) * on

        if ftype == "LG":
            i_extra[p] += fault_wave(cfg.v_ln, PHASE_SHIFT[p])
            phases = [p]
        elif ftype == "LL":
            q = (p + 1) % 3
            ang = np.angle(np.exp(1j * PHASE_SHIFT[p]) - np.exp(1j * PHASE_SHIFT[q]))
            w = fault_wave(np.sqrt(3) * cfg.v_ln, ang) / 2.0   # loop through two conductors
            i_extra[p] += w
            i_extra[q] -= w
            phases = [p, q]
        else:
            for k in range(3):
                i_extra[k] += fault_wave(cfg.v_ln, PHASE_SHIFT[k])
            phases = [0, 1, 2]
        meta.update(fault_type=str(ftype), phases=[PHASES[k] for k in phases],
                    fault_rms=round(cfg.v_ln / z, 1), tau_ms=round(tau * 1e3, 2))

    def _break_phase(self, rng, t, t0, i_load, p):
        """Phase p conductor parts at t0: only loads upstream of the break stay connected."""
        upstream = rng.uniform(0.0, 0.35)
        drop = 1 - (1 - upstream) * _step(t, t0, rise=0.001)
        i_load[p] *= drop
        return upstream

    def _ev_open_conductor(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        p = int(rng.integers(3))
        upstream = self._break_phase(rng, t, t0, i_load, p)
        # tiny capacitive charging current of the dangling section, leads voltage by 90 deg
        cap = rng.uniform(0.05, 0.4)
        i_extra[p] += np.sqrt(2) * cap * np.cos(wt + PHASE_SHIFT[p]) * (t >= t0)
        meta.update(phase=PHASES[p], upstream_load_fraction=round(upstream, 3))

    def _hif_current(self, rng, t, v, t0, r_lo, r_hi):
        """Two-diode (Emanuel) arc model with per-half-cycle randomness.

        Conducts only when |v| exceeds the arc voltage. Positive and negative
        half cycles use different arc voltages and resistances (asymmetry ->
        even harmonics and DC). A build-up period models the conductor settling
        and moisture boiling off; random half cycles extinguish completely.
        """
        cfg = self.cfg
        r0 = rng.uniform(r_lo, r_hi)
        vp0 = rng.uniform(40, 110)                       # arc voltage, positive half cycles
        vn0 = vp0 * rng.uniform(0.75, 1.25)              # asymmetric negative half cycles
        buildup = rng.uniform(0.15, 1.0)                 # seconds
        extinguish_p = rng.uniform(0.02, 0.20)
        i = np.zeros_like(t)
        half = cfg.spc // 2
        start = int(np.searchsorted(t, t0))
        for s in range(start, len(t), half):
            e = min(s + half, len(t))
            if rng.random() < extinguish_p:
                continue
            age = t[s] - t0
            settle = 1 + 2.5 * np.exp(-age / (buildup / 3))   # resistance falls as contact improves
            r = r0 * settle * rng.uniform(0.85, 1.15)
            vp = vp0 * rng.uniform(0.9, 1.1)
            vn = vn0 * rng.uniform(0.9, 1.1)
            seg = v[s:e]
            pos = seg > vp
            neg = seg < -vn
            i[s:e][pos] = (seg[pos] - vp) / r
            i[s:e][neg] = (seg[neg] + vn) / (r * rng.uniform(0.9, 1.1))
        return i, dict(surface_r=round(r0, 1), vp=round(vp0, 1), vn=round(vn0, 1),
                       buildup_s=round(buildup, 3), extinguish_p=round(extinguish_p, 3))

    def _ev_downed_conductor(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        p = int(rng.integers(3))
        upstream = self._break_phase(rng, t, t0, i_load, p)
        surface = str(rng.choice(list(SURFACES)))
        r_lo, r_hi = SURFACES[surface]
        source_end_down = rng.random() < 0.75
        if not source_end_down:
            # load-side end on the ground: only back-fed through loads, much weaker
            r_lo, r_hi = r_lo * 4, r_hi * 4
        i_hif, hif_meta = self._hif_current(rng, t, vs[p], t0, r_lo, r_hi)
        i_extra[p] += i_hif
        on = t >= t0 + 0.5
        meta.update(phase=PHASES[p], surface=surface, source_end_down=source_end_down,
                    upstream_load_fraction=round(upstream, 3),
                    hif_rms_after_settle=round(float(np.sqrt((i_hif[on] ** 2).mean())), 3),
                    **hif_meta)

    def _ev_series_arc(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        """Loose joint arcing in series with the load on one phase.

        After each current zero the arc needs time to re-strike, so current
        stays near zero for a random 'shoulder' (0.2-1.5 ms), and conduction
        carries broadband noise from the unstable arc.
        """
        cfg = self.cfg
        p = int(rng.integers(3))
        x = i_load[p].copy()
        on = t >= t0
        sign = np.sign(x)
        zc = np.where((np.diff(sign) != 0) & on[1:])[0] + 1
        mask = np.ones_like(x)
        for z in zc:
            d = int(rng.uniform(0.2e-3, 1.5e-3) * cfg.fs)
            mask[z:z + d] = 0.0
        noise_level = rng.uniform(0.01, 0.05)
        noise = rng.normal(0, 1, len(t))
        noise = np.convolve(noise, np.ones(3) / 3, mode="same")   # broadband, not white
        amp = np.sqrt(2) * np.sqrt((x ** 2).mean())
        flicker = np.repeat(rng.uniform(0.92, 1.0, len(t) // (cfg.spc // 2) + 1), cfg.spc // 2)[:len(t)]
        arced = x * mask * flicker + noise_level * amp * noise * mask
        i_load[p] = np.where(on, arced, x)
        meta.update(phase=PHASES[p], noise_level=round(noise_level, 3))

    # ---- look-alike disturbances (label Normal) ---------------------------------------------
    def _ev_motor_start(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        im = rng.uniform(10, 45)                         # motor rated current, A
        k = rng.uniform(5, 7)                            # locked-rotor multiple
        tau = rng.uniform(0.15, 0.9)                     # acceleration time constant
        on = (t >= t0)
        decay = np.exp(-(t - t0) / tau) * on
        mag = im * (1 + (k - 1) * decay) * on
        phi = np.arccos(0.3) * decay + np.arccos(0.85) * (1 - decay)   # pf improves as it speeds up
        for kph in range(3):
            i_extra[kph] += np.sqrt(2) * mag * np.sin(wt + PHASE_SHIFT[kph] - phi)
        meta.update(motor_rated=round(im, 1), inrush_multiple=round(k, 2), tau_s=round(tau, 3))

    def _ev_capacitor_switching(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        cfg = self.cfg
        q = rng.uniform(15e3, 75e3)                      # kVAr bank
        ic = q / (3 * cfg.v_ln)
        fr = rng.uniform(300, 900)                       # resonant ringing frequency
        tau = rng.uniform(0.002, 0.008)
        on = (t >= t0)
        ring = np.exp(-np.clip(t - t0, 0, None) / tau) * np.sin(2 * np.pi * fr * (t - t0)) * on
        # LT banks are delta-connected: inrush depends on each phase's voltage at the
        # switching instant, and the three phase currents sum to zero (no neutral current).
        f = meta["frequency"]
        v_at_switch = np.sin(2 * np.pi * f * t0 + PHASE_SHIFT)
        v_at_switch -= v_at_switch.mean()
        inrush = rng.uniform(3, 10)
        v_ring = rng.uniform(0.15, 0.5)
        for k in range(3):
            i_extra[k] += np.sqrt(2) * ic * np.cos(wt + PHASE_SHIFT[k]) * on        # leading current
            i_extra[k] += np.sqrt(2) * ic * inrush * v_at_switch[k] * ring
            v_extra[k] -= np.sqrt(2) * cfg.v_ln * v_ring * v_at_switch[k] * ring
        meta.update(kvar=round(q / 1e3, 1), ring_hz=round(fr, 1))

    def _ev_load_step(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        # a look-alike must stay a normal condition: never push the feeder past 95% of rating
        cur = np.sqrt((i_load ** 2).mean(axis=1)).max()
        headroom = 0.95 * self.cfg.rated_current / cur - 1
        frac = rng.uniform(0.15, 0.4) * rng.choice([-1, 1])
        if frac > 0:
            frac = min(frac, max(headroom, 0.05)) if headroom > 0.05 else -frac
        i_load *= (1 + frac * _step(t, t0, rise=0.01))[None]
        meta.update(step_fraction=round(frac, 3))

    def _ev_single_phase_load(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        """A big single-phase load switching on raises the residual current
        just like a ground fault, but cleanly: no arcing randomness."""
        p = int(rng.integers(3))
        amp = rng.uniform(15, 60)
        pf = np.arccos(rng.uniform(0.85, 1.0))
        i_extra[p] += np.sqrt(2) * amp * np.sin(wt + PHASE_SHIFT[p] - pf) * _step(t, t0, 0.005)
        meta.update(phase=PHASES[p], added_rms=round(amp, 1))

    def _ev_nonlinear_load(self, rng, t, wt, t0, vs, i_load, i_extra, v_extra, meta):
        """Inverters / LED drivers / SMPS switching on: harmonic-rich current,
        which raises the 3rd harmonic that many HIF detectors rely on."""
        p = int(rng.integers(3))
        amp = rng.uniform(10, 50)
        h3 = rng.uniform(0.4, 0.8)
        h5 = rng.uniform(0.2, 0.5)
        theta = wt + PHASE_SHIFT[p]
        wave = np.sin(theta) + h3 * np.sin(3 * theta + rng.uniform(0, 2 * np.pi)) \
            + h5 * np.sin(5 * theta + rng.uniform(0, 2 * np.pi))
        i_extra[p] += np.sqrt(2) * amp * wave * _step(t, t0, 0.01)
        meta.update(phase=PHASES[p], added_rms=round(amp, 1), h3=round(h3, 2))


def simulate(kind: str, seed: int | None = None, cfg: FeederConfig | None = None) -> Event:
    """Convenience wrapper: simulate one event."""
    return FeederSimulator(cfg).simulate(kind, seed)
