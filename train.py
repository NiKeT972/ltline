"""Train the two LTLine models and save them to ./models.

    python train.py

1. Isolation Forest, trained only on healthy readings  -> early-warning anomaly score
2. Random Forest over 5 classes                         -> fault type + confidence

Each class segment is split in time order (first 80% train, last 20% test),
so the test set is never interleaved with training samples.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ltline.core import (CLASSES, FAULTS, FEATURES, MODELS_DIR, WINDOW, load_segments,
                         make_features, simulate_trip)

TEST_FRAC = 0.2


TRANSITIONS_PER_FAULT = 60
SCENARIOS_PER_FAULT = 100


def onset_samples(healthy: pd.DataFrame, fault: pd.DataFrame, label: str,
                  rng: np.random.Generator) -> pd.DataFrame:
    """Features for the first few samples after a fault begins.

    The rolling window then spans healthy and fault readings. Without these
    examples the step change at onset looks 'jagged' and gets confused with Arc.
    """
    rows = []
    for _ in range(TRANSITIONS_PER_FAULT):
        h0 = rng.integers(0, len(healthy) - WINDOW)
        f0 = rng.integers(0, len(fault) - 2 * WINDOW)
        stream = pd.concat([healthy.iloc[h0:h0 + WINDOW], fault.iloc[f0:f0 + 2 * WINDOW]],
                           ignore_index=True)
        f = make_features(stream).iloc[WINDOW:]
        rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    out["label"] = label
    return out


def evaluate_scenarios(parts, iso, rf, threshold, rng):
    """End-to-end check on held-out data: healthy stretch, then a fault.

    Counts correct trips, wrong-type trips, false trips (before the fault starts),
    missed faults, and the delay in samples from fault onset to trip.
    """
    healthy = parts["Healthy"][1]
    per_fault, delays = {}, []
    false_trips = 0
    for fault in FAULTS:
        seg = parts[fault][1]
        counts = {"correct": 0, "wrong_type": 0, "missed": 0}
        for _ in range(SCENARIOS_PER_FAULT):
            h_len = int(rng.integers(60, 140))
            f_len = min(40, len(seg))
            h0 = int(rng.integers(0, len(healthy) - h_len))
            f0 = int(rng.integers(0, len(seg) - f_len + 1))
            stream = pd.concat([healthy.iloc[h0:h0 + h_len], seg.iloc[f0:f0 + f_len]],
                               ignore_index=True)
            trip = simulate_trip(stream, iso, rf, threshold)
            if trip is None:
                counts["missed"] += 1
            elif trip[0] < h_len:
                false_trips += 1
            elif trip[1] == fault:
                counts["correct"] += 1
                delays.append(trip[0] - h_len)
            else:
                counts["wrong_type"] += 1
        per_fault[fault] = counts
    total = SCENARIOS_PER_FAULT * len(FAULTS)
    correct = sum(c["correct"] for c in per_fault.values())
    return {
        "scenarios": total,
        "correct_trip_rate": correct / total,
        "false_trips": false_trips,
        "median_trip_delay_samples": float(np.median(delays)) if delays else None,
        "per_fault": per_fault,
    }


def build_split():
    rng = np.random.default_rng(42)
    train, test, parts = [], [], {}
    for label, seg in load_segments().items():
        cut = int(len(seg) * (1 - TEST_FRAC))
        parts[label] = (seg.iloc[:cut].reset_index(drop=True), seg.iloc[cut:].reset_index(drop=True))
        f = make_features(seg)
        f["label"] = label
        train.append(f.iloc[:cut])
        test.append(f.iloc[cut:])
    # Onset examples, built separately from the train and test portions so they never mix
    for label in FAULTS:
        train.append(onset_samples(parts["Healthy"][0], parts[label][0], label, rng))
        test.append(onset_samples(parts["Healthy"][1], parts[label][1], label, rng))
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True), parts


def main() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    train, test, parts = build_split()
    X_tr, y_tr = train[FEATURES], train["label"]
    X_te, y_te = test[FEATURES], test["label"]

    # Anomaly detector: learns what "normal" looks like, no fault labels needed.
    healthy_tr = X_tr[y_tr == "Healthy"]
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(healthy_tr)
    healthy_scores = -iso.decision_function(healthy_tr)
    threshold = float(np.percentile(healthy_scores, 99))

    scores_te = -iso.decision_function(X_te)
    flagged = scores_te > threshold
    is_fault = (y_te != "Healthy").to_numpy()
    anomaly_recall = float(flagged[is_fault].mean())
    false_alarm_rate = float(flagged[~is_fault].mean())

    # Fault classifier
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    report = classification_report(y_te, pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=CLASSES)

    scen = evaluate_scenarios(parts, iso, rf, threshold, np.random.default_rng(7))

    dump(iso, MODELS_DIR / "isoforest.joblib", compress=3)
    dump(rf, MODELS_DIR / "fault_rf.joblib", compress=3)
    meta = {
        "classes": CLASSES,
        "features": FEATURES,
        "default_anomaly_threshold": threshold,
        "test_accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "anomaly_recall_on_faults": anomaly_recall,
        "anomaly_false_alarm_rate": false_alarm_rate,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "scenario_eval": scen,
    }
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    (MODELS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(MODELS_DIR / "confusion_matrix.csv")

    print(f"Test accuracy:      {acc:.3f}")
    print(f"Macro F1:           {meta['macro_f1']:.3f}")
    print(f"Anomaly recall:     {anomaly_recall:.3f}  (faults flagged by Isolation Forest)")
    print(f"False-alarm rate:   {false_alarm_rate:.3f}  (healthy flagged)")
    print(f"Scenario test:      {scen['correct_trip_rate']:.1%} correct trips over "
          f"{scen['scenarios']} held-out streams, {scen['false_trips']} false trips, "
          f"median delay {scen['median_trip_delay_samples']} samples")
    print("  per fault:", scen["per_fault"])
    print(classification_report(y_te, pred, labels=CLASSES, digits=3, zero_division=0))
    return meta


if __name__ == "__main__":
    main()
