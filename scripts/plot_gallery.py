"""Plot one example of every event kind: phase currents, residual current and voltages
around the event onset. Saves docs/waveform_gallery.png."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ltline.sim.feeder import ALL_KINDS, FeederSimulator  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "waveform_gallery.png"
COL = {"a": "#c0392b", "b": "#d4a017", "c": "#1f4e9c"}


def main(seed=3):
    sim = FeederSimulator()
    fig, axes = plt.subplots(len(ALL_KINDS), 3, figsize=(16, 2.1 * len(ALL_KINDS)))
    for r, kind in enumerate(ALL_KINDS):
        ev = sim.simulate(kind, seed + r)
        t = ev.t
        t0 = ev.onset or 1.2
        w = (t > t0 - 0.06) & (t < t0 + 0.14)
        tt = (t[w] - t0) * 1e3
        s = ev.signals
        for k, ph in enumerate("abc"):
            axes[r, 0].plot(tt, s[3 + k][w], color=COL[ph], lw=0.9, label=f"I{ph}")
            axes[r, 2].plot(tt, s[k][w], color=COL[ph], lw=0.9)
        res = s[3] + s[4] + s[5]
        axes[r, 1].plot(tt, res[w], color="#333", lw=0.9)
        if ev.label == "DownedConductor":
            p = "abc".index(ev.meta["phase"])
            axes[r, 1].set_title(f"HIF current ~{ev.meta['hif_rms_after_settle']} A rms on {ev.meta['surface']}", fontsize=8)
        axes[r, 0].set_ylabel(f"{kind}\n[{ev.label}]", fontsize=9)
        for c in range(3):
            axes[r, c].axvline(0, color="grey", ls="--", lw=0.7)
            axes[r, c].tick_params(labelsize=7)
    axes[0, 0].set_title("Phase currents (A)")
    axes[0, 1].set_title("Residual / neutral current Ia+Ib+Ic (A)")
    axes[0, 2].set_title("Bus voltages (V)")
    for c in range(3):
        axes[-1, c].set_xlabel("ms from event onset")
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
