"""Seeded, reproducible event datasets.

Raw waveforms are large (6 channels x 9,600 samples per event), but the
simulator is deterministic and fast (~6 ms per event). So a dataset is stored
as a small manifest of (event_id, kind, seed, label, onset, split, metadata),
and waveforms are regenerated on demand. Same manifest + same code = same data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .feeder import DISTURBANCE_KINDS, FAULT_KINDS, Event, FeederConfig, FeederSimulator

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "sim_manifest.csv"


def build_manifest(n_per_fault: int = 400, n_per_disturbance: int = 200,
                   seed: int = 2025, test_frac: float = 0.2,
                   cfg: FeederConfig | None = None) -> pd.DataFrame:
    """Simulate every event once to record its label, onset and metadata.

    The split is decided per event before anything is learned, and test events
    use their own seeds, so no waveform appears in both train and test.
    """
    sim = FeederSimulator(cfg)
    master = np.random.default_rng(seed)
    plan = [(k, n_per_fault) for k in FAULT_KINDS] + [(k, n_per_disturbance) for k in DISTURBANCE_KINDS]
    rows = []
    for kind, n in plan:
        seeds = master.choice(2**31 - 1, size=n, replace=False)
        n_test = int(round(n * test_frac))
        for j, s in enumerate(seeds):
            ev = sim.simulate(kind, int(s))
            rows.append({
                "event_id": f"{kind}-{j:04d}",
                "kind": kind,
                "seed": int(s),
                "label": ev.label,
                "onset": ev.onset,
                "split": "test" if j < n_test else "train",
                "meta": json.dumps(ev.meta, default=_json_default),
            })
    return pd.DataFrame(rows)


def load_event(row, cfg: FeederConfig | None = None) -> Event:
    """Regenerate the waveforms for one manifest row."""
    return FeederSimulator(cfg).simulate(row["kind"], int(row["seed"]))


def iter_events(manifest: pd.DataFrame, cfg: FeederConfig | None = None):
    sim = FeederSimulator(cfg)
    for _, row in manifest.iterrows():
        yield row, sim.simulate(row["kind"], int(row["seed"]))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))
