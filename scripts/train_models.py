"""Train and compare protection approaches on the simulated feeder.

    python scripts/train_models.py

Approaches
  1. conventional : overcurrent + earth-fault relay settings (what's on poles today)
  2. rms_rf       : v1-style ML - Random Forest on RMS current/voltage only
  3. rf           : Random Forest on all relay-style features
  4. lgbm         : LightGBM on all relay-style features

Operating point (probability threshold, K consecutive cycles) is tuned on a
validation split of the TRAIN events - the test events are only touched once,
at the end. Results -> models/v2/results.json, models/v2/events_test.csv
"""
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltline.protection import evaluate_events, summarise, tune  # noqa: E402

OUT = ROOT / "models" / "v2"
META_COLS = ["cycle", "t", "event_id", "kind", "label", "split", "since_onset", "y"]
RMS_FEATURES = ["i_max", "i1", "i_spread", "v_min_pu", "v1_pu"]
RATED = 200.0

# conventional settings (typical LT feeder pillar / ACB release)
OC_INST = 2.0        # x rated, instantaneous short-circuit element
OC_THERMAL = 1.1     # x rated, overload element
OC_THERMAL_CYC = 25  # cycles above OC_THERMAL before trip (0.5 s, simplified inverse-time)
EF_PICKUP = 60.0     # A residual (0.3 x rated: lower would nuisance-trip on normal unbalance)


def conventional(win: pd.DataFrame, ef_pickup: float = EF_PICKUP,
                 thermal_cyc: int = OC_THERMAL_CYC) -> pd.DataFrame:
    rows = []
    for eid, w in win.groupby("event_id", sort=False):
        trip, cls, j = False, None, None
        over = 0
        for i, (imax, res1) in enumerate(zip(w.i_max.to_numpy(), w.res1.to_numpy())):
            if imax > OC_INST:
                trip, cls, j = True, "ShortCircuit", i
                break
            over = over + 1 if imax > OC_THERMAL else 0
            if over >= thermal_cyc:
                trip, cls, j = True, "Overload", i
                break
            if res1 > ef_pickup:
                trip, cls, j = True, "EarthFault", i
                break
        label = w.label.iat[0]
        r = {"event_id": eid, "kind": w.kind.iat[0], "label": label, "tripped": trip,
             "trip_class": cls, "delay_ms": np.nan, "early": False}
        if trip and label != "Normal":
            so = w.since_onset.iat[j]
            r["early"] = so <= 0
            r["delay_ms"] = so * 1000 if so > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    feats = [c for c in df.columns if c not in META_COLS]
    df = df.sort_values(["event_id", "cycle"]).reset_index(drop=True)

    train_ev = df.loc[df.split == "train", "event_id"].unique()
    rng = np.random.default_rng(0)
    val_ev = set(rng.choice(train_ev, size=int(0.2 * len(train_ev)), replace=False))
    is_val = df.event_id.isin(val_ev)
    fit = df[(df.split == "train") & ~is_val & df.y.notna()]
    val = df[(df.split == "train") & is_val]
    test = df[df.split == "test"]
    print(f"fit windows {len(fit):,} | val events {val.event_id.nunique()} | test events {test.event_id.nunique()}")

    results = {}

    # 1. conventional relay - earth-fault pickup set like an engineer would: the lowest
    #    setting that does not nuisance-trip on the validation events
    best = None
    for thermal in [25, 50, 75]:
        for pickup in [40, 50, 60, 80, 100, 120, 150, 200, 250, 300]:
            s_val = summarise(conventional(val, pickup, thermal))
            fa = s_val["false_trips_normal"] + s_val["false_trips_early"]
            score = (-fa, s_val["dependability"], -pickup, -thermal)
            if best is None or score > best[0]:
                best = (score, pickup, thermal)
    _, ef, thermal = best
    ev = conventional(test, ef, thermal)
    results["conventional"] = {"summary": summarise(ev), "settings": {
        "inst_x_rated": OC_INST, "overload_x_rated": OC_THERMAL,
        "overload_cycles": thermal, "earth_fault_A": ef}}
    ev.assign(model="conventional").to_csv(OUT / "events_test_conventional.csv", index=False)

    # 2-4. ML models
    def make(name):
        if name == "rms_rf":
            return RandomForestClassifier(200, max_depth=14, class_weight="balanced",
                                          n_jobs=-1, random_state=0), RMS_FEATURES
        if name == "rf":
            return RandomForestClassifier(300, max_depth=18, min_samples_leaf=3,
                                          class_weight="balanced", n_jobs=-1, random_state=0), feats
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                  class_weight="balanced", subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=0, verbose=-1), feats

    for name in ["rms_rf", "rf", "lgbm"]:
        model, cols = make(name)
        model.fit(fit[cols], fit.y)
        classes = list(model.classes_)
        th, k, val_s = tune(val, model.predict_proba(val[cols]), classes)
        ev = evaluate_events(test, model.predict_proba(test[cols]), classes, th, k)
        s = summarise(ev)
        results[name] = {"summary": s, "threshold": th, "k_cycles": k, "val": val_s, "features": cols}
        ev.assign(model=name).to_csv(OUT / f"events_test_{name}.csv", index=False)
        joblib.dump({"model": model, "features": cols, "classes": classes,
                     "threshold": th, "k": k}, OUT / f"{name}.joblib", compress=3)
        print(f"{name:7s} th={th} K={k}  dependability {s['dependability']:.3f}  "
              f"correct {s['correct_type']:.3f}  broken {s['broken_conductor_caught']:.3f}  "
              f"false trips {s['false_trips_normal']}+{s['false_trips_early']}  "
              f"delay {s['median_delay_ms']:.0f} ms")

    s = results["conventional"]["summary"]
    print(f"conventional (EF {ef} A) dependability {s['dependability']:.3f}  correct {s['correct_type']:.3f}  "
          f"false trips {s['false_trips_normal']}+{s['false_trips_early']}")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
