"""Trip logic and event-level evaluation shared by training, the app and reports.

A protection decision is made per event (a stream of windows), not per window:
the isolator trips when the same fault class wins with probability >= threshold
for K consecutive windows (one window per 20 ms cycle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAULTS = ["Overload", "ShortCircuit", "OpenConductor", "DownedConductor", "SeriesArc"]
BROKEN = {"OpenConductor", "DownedConductor"}


FAST = {"ShortCircuit"}   # cleared fast, like an instantaneous element
K_FAST = 2


def trip_decision(proba: np.ndarray, classes: list[str], threshold: float, k: int):
    """proba: (n_windows, n_classes) in time order. Returns (window_index, class) or None.

    Short circuits trip after K_FAST consecutive cycles; everything else after k.
    """
    best = proba.argmax(axis=1)
    pbest = proba[np.arange(len(proba)), best]
    run_cls, run = None, 0
    for i, (b, p) in enumerate(zip(best, pbest)):
        cls = classes[b]
        if cls != "Normal" and p >= threshold:
            run = run + 1 if cls == run_cls else 1
            run_cls = cls
            if run >= (K_FAST if cls in FAST else k):
                return i, cls
        else:
            run_cls, run = None, 0
    return None


def evaluate_events(windows: pd.DataFrame, proba: np.ndarray, classes: list[str],
                    threshold: float, k: int) -> pd.DataFrame:
    """One row per event: what tripped, when, and whether it was right.

    windows must hold event_id, kind, label, t, since_onset in time order per event.
    """
    rows = []
    windows = windows.reset_index(drop=True)
    for eid, idx in windows.groupby("event_id", sort=False).indices.items():
        w = windows.iloc[idx]
        d = trip_decision(proba[idx], classes, threshold, k)
        label = w["label"].iat[0]
        r = {"event_id": eid, "kind": w["kind"].iat[0], "label": label,
             "tripped": d is not None, "trip_class": None, "delay_ms": np.nan, "early": False}
        if d is not None:
            j, cls = d
            r["trip_class"] = cls
            so = w["since_onset"].iat[j]
            if label != "Normal":
                if so <= 0:
                    r["early"] = True            # tripped before the fault existed: a false trip
                else:
                    r["delay_ms"] = so * 1000
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame) -> dict:
    """Dependability (faults caught), selectivity (right type) and security (no false trips)."""
    faults = ev[ev.label != "Normal"]
    normal = ev[ev.label == "Normal"]
    caught = faults.tripped & ~faults.early
    correct = caught & (faults.trip_class == faults.label)
    broken = faults.label.isin(BROKEN)
    broken_ok = caught & broken & faults.trip_class.isin(BROKEN)
    out = {
        "fault_events": int(len(faults)),
        "dependability": float(caught.mean()),                       # fault events that tripped
        "correct_type": float(correct.mean()),
        "broken_conductor_caught": float(broken_ok[broken].mean()) if broken.any() else np.nan,
        "normal_events": int(len(normal)),
        "false_trips_normal": int(normal.tripped.sum()),
        "false_trips_early": int(faults.early.sum()),
        "security": float(1 - normal.tripped.mean()),
        "median_delay_ms": float(faults.loc[correct, "delay_ms"].median()) if correct.any() else np.nan,
    }
    per = {}
    for lab, g in faults.groupby("label"):
        c = g.tripped & ~g.early
        per[lab] = {"caught": float(c.mean()),
                    "correct_type": float((c & (g.trip_class == lab)).mean()),
                    "median_delay_ms": float(g.loc[c & (g.trip_class == lab), "delay_ms"].median())}
    for kind, g in normal.groupby("kind"):
        per[f"Normal/{kind}"] = {"false_trip_rate": float(g.tripped.mean()), "n": int(len(g))}
    out["per_class"] = per
    return out


def tune(windows: pd.DataFrame, proba: np.ndarray, classes: list[str]):
    """Pick (threshold, K) on validation events: zero false trips first, then the
    most faults tripped with the correct type, then the most faults tripped, then faster."""
    best = None
    for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for k in [2, 3, 4, 6, 8, 12]:
            s = summarise(evaluate_events(windows, proba, classes, th, k))
            fa = s["false_trips_normal"] + s["false_trips_early"]
            score = (fa == 0, s["correct_type"], s["dependability"], -k)
            if best is None or score > best[0]:
                best = (score, th, k, s)
    return best[1], best[2], best[3]
