"""Build the Phase 2 results: comparison table (markdown) and figures.

    python scripts/report.py      # after train_models.py and train_cnn.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "models" / "v2"
DOCS = ROOT / "docs"
NAMES = {"conventional": "Conventional relay (OC + EF)", "rms_rf": "v1-style ML (RMS only)",
         "rf": "Random Forest (relay features)", "lgbm": "LightGBM (relay features)",
         "cnn": "1D-CNN (raw waveforms)"}
FAULTS = ["ShortCircuit", "Overload", "OpenConductor", "DownedConductor", "SeriesArc"]
BLUE, GREY, RED = "#1f4e9c", "#9aa5b1", "#b42318"


def main():
    res = json.loads((OUT / "results.json").read_text())
    models = [m for m in NAMES if m in res]
    rows = []
    for m in models:
        s = res[m]["summary"]
        pc = s["per_class"]
        row = {"Approach": NAMES[m],
               "Faults tripped": s["dependability"],
               "Right fault type": s["correct_type"],
               "Broken conductor caught": s["broken_conductor_caught"],
               "False trips (of 240 normal)": s["false_trips_normal"] + s["false_trips_early"],
               "SC trip time (ms)": pc.get("ShortCircuit", {}).get("median_delay_ms", np.nan)}
        for f in FAULTS:
            row[f] = pc.get(f, {}).get("caught", np.nan)
        rows.append(row)
    tab = pd.DataFrame(rows)

    def pct(x):
        return "-" if pd.isna(x) else f"{x:.1%}"
    md = ["| Approach | Faults tripped | Right type | Broken conductor caught | False trips | SC trip time |",
          "|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        sc = "-" if pd.isna(r["SC trip time (ms)"]) else f"{r['SC trip time (ms)']:.0f} ms"
        md.append(f"| {r.Approach} | {pct(r['Faults tripped'])} | {pct(r['Right fault type'])} | "
                  f"{pct(r['Broken conductor caught'])} | {r['False trips (of 240 normal)']} / 240 | {sc} |")
    md += ["", "Faults tripped, per class:", "",
           "| Approach | " + " | ".join(FAULTS) + " |", "|---" * (len(FAULTS) + 1) + "|"]
    for _, r in tab.iterrows():
        md.append(f"| {r.Approach} | " + " | ".join(pct(r[f]) for f in FAULTS) + " |")
    (OUT / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- figure 1: per-class detection, conventional vs best
    best = max([m for m in models if m != "conventional"],
               key=lambda m: (res[m]["summary"]["false_trips_normal"] == 0, res[m]["summary"]["dependability"]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(FAULTS))
    conv = tab.set_index("Approach").loc[NAMES["conventional"], FAULTS].astype(float)
    bst = tab.set_index("Approach").loc[NAMES[best], FAULTS].astype(float)
    ax.bar(x - 0.2, conv * 100, 0.4, color=GREY, label=NAMES["conventional"])
    ax.bar(x + 0.2, bst * 100, 0.4, color=BLUE, label=NAMES[best])
    for i, (a, b) in enumerate(zip(conv, bst)):
        ax.text(i - 0.2, a * 100 + 2, f"{a:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.2, b * 100 + 2, f"{b:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Short circuit", "Overload", "Open conductor", "Downed conductor\n(HIF)", "Series arc"])
    ax.set_ylabel("Fault events tripped (%)")
    ax.set_ylim(0, 115)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(DOCS / "detection_by_fault.png", dpi=130)

    # ---- figure 2: downed-conductor detection by surface and fault current
    man = pd.read_csv(ROOT / "data" / "sim_manifest.csv")
    meta = man.set_index("event_id").meta.map(json.loads)
    ev = pd.read_csv(OUT / f"events_test_{best}.csv")
    d = ev[ev.label == "DownedConductor"].copy()
    d["surface"] = d.event_id.map(lambda e: meta[e]["surface"])
    d["hif_a"] = d.event_id.map(lambda e: meta[e]["hif_rms_after_settle"])
    d["caught"] = d.tripped & ~d.early
    d["as_downed"] = d.caught & (d.trip_class == "DownedConductor")
    d["band"] = pd.cut(d.hif_a, [-0.01, 0.2, 0.5, 1, 2, 100],
                       labels=["<0.2 A", "0.2-0.5 A", "0.5-1 A", "1-2 A", ">2 A"])
    by_band = d.groupby("band", observed=False)[["caught", "as_downed"]].mean()
    by_band["n"] = d.groupby("band", observed=False).size()
    by_surf = d.groupby("surface")[["caught", "as_downed"]].mean()
    by_surf["n"] = d.groupby("surface").size()
    print("\nDowned conductor by fault current:\n", by_band.round(2))
    print("\nDowned conductor by surface:\n", by_surf.round(2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, tb, title in [(axes[0], by_band, "By fault current to ground"),
                          (axes[1], by_surf.sort_values("caught"), "By ground surface")]:
        xi = np.arange(len(tb))
        ax.bar(xi - 0.2, tb.caught * 100, 0.4, color=BLUE, label="Line tripped")
        ax.bar(xi + 0.2, tb.as_downed * 100, 0.4, color="#7fa7e0", label="...and identified as on the ground")
        ax.set_xticks(xi, [f"{i}\n(n={n})" for i, n in zip(tb.index.astype(str).str.replace('_', ' '), tb.n)], fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Downed-conductor events (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(DOCS / "hif_detection.png", dpi=130)

    # ---- figure 3: feature importance (LightGBM gain)
    if "lgbm" in res:
        b = joblib.load(OUT / "lgbm.joblib")
        imp = pd.Series(b["model"].booster_.feature_importance("gain"), index=b["features"])
        imp = (imp / imp.sum()).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.barh(imp.index, imp.values * 100, color=BLUE)
        ax.set_xlabel("Share of total gain (%)")
        ax.set_title("What LightGBM relies on (top 15 features)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(DOCS / "feature_importance.png", dpi=130)
        print("\nTop features:", imp.sort_values(ascending=False).round(3).to_dict())

    summary = {"best_model": best, "downed_by_current": by_band.reset_index().astype({"band": str}).to_dict("records"),
               "downed_by_surface": by_surf.reset_index().to_dict("records")}
    (OUT / "hif_breakdown.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
