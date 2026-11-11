"""Compute window features for every event in data/sim_manifest.csv.

    python scripts/build_features.py          # ~3 min on 2 cores

Writes data/features.parquet: one row per (event, window end cycle), with
event_id, kind, label, split, cycle, t (seconds), phase ('pre'/'post'), y (training label).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltline.features import event_features  # noqa: E402
from ltline.sim.feeder import FeederSimulator  # noqa: E402

DETECT_CYCLES = 35   # windows labelled as the event for this long after onset


def one(row):
    ev = FeederSimulator().simulate(row["kind"], int(row["seed"]))
    df, f0 = event_features(ev.signals, ev.fs)
    df = df.reset_index()
    df["t"] = (df["cycle"] + 1) / f0
    df["event_id"], df["kind"], df["label"], df["split"] = row["event_id"], row["kind"], row["label"], row["split"]
    onset = row["onset"]
    if pd.isna(onset):
        df["since_onset"] = np.nan
    else:
        df["since_onset"] = df["t"] - onset
    # training target: Normal before onset; the event label inside the detection zone;
    # windows after the zone are kept for stream evaluation but not used for training (y = None)
    df["y"] = "Normal"
    if row["label"] != "Normal":
        so = df["since_onset"]
        zone = (so > 0) & (so <= DETECT_CYCLES / f0)
        df.loc[zone, "y"] = row["label"]
        df.loc[so > DETECT_CYCLES / f0, "y"] = None
    return df


def main():
    m = pd.read_csv(ROOT / "data" / "sim_manifest.csv")
    with Pool() as pool:
        parts = pool.map(one, [r for _, r in m.iterrows()], chunksize=20)
    feats = pd.concat(parts, ignore_index=True)
    out = ROOT / "data" / "features.parquet"
    feats.to_parquet(out, index=False)
    print(f"{len(feats):,} windows x {feats.shape[1]} columns -> {out}")
    print(feats.groupby(["split", "y"], dropna=False).size().unstack(0))


if __name__ == "__main__":
    main()
