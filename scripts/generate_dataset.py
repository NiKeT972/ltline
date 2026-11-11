"""Build the simulated LT-feeder dataset manifest (data/sim_manifest.csv).

    python scripts/generate_dataset.py                 # 2,000 faults + 1,200 normal events
    python scripts/generate_dataset.py --per-fault 50  # quick small set

Waveforms are regenerated on demand from each row's seed (see ltline/sim/dataset.py).
Pass --export-npz to also write raw waveforms (about 230 KB per event).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ltline.sim.dataset import MANIFEST_PATH, build_manifest, iter_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fault", type=int, default=400)
    ap.add_argument("--per-disturbance", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--export-npz", type=Path, default=None,
                    help="optional folder to write raw waveforms as .npz files")
    args = ap.parse_args()

    m = build_manifest(args.per_fault, args.per_disturbance, args.seed)
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    m.to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {len(m)} events to {MANIFEST_PATH}")
    print(m.groupby(["label", "split"]).size().unstack())

    hif = m[m.kind == "downed_conductor"].meta.map(json.loads)
    amps = np.array([h["hif_rms_after_settle"] for h in hif])
    print(f"\nDowned-conductor fault current (rms): median {np.median(amps):.2f} A, "
          f"90th pct {np.percentile(amps, 90):.2f} A, max {amps.max():.2f} A "
          f"(feeder rating 200 A)")
    print("Surfaces:", hif.map(lambda h: h["surface"]).value_counts().to_dict())

    if args.export_npz:
        args.export_npz.mkdir(parents=True, exist_ok=True)
        for row, ev in iter_events(m):
            np.savez_compressed(args.export_npz / f"{row.event_id}.npz", signals=ev.signals,
                                fs=ev.fs, label=ev.label, onset=ev.onset or -1)
        print("Exported waveforms to", args.export_npz)


if __name__ == "__main__":
    main()
