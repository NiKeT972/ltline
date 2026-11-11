"""1D-CNN on raw waveforms - no hand-made features.

    python scripts/train_cnn.py

Input per window (4 cycles = 256 samples after frequency-tracked resampling):
  7 channels  Va Vb Vc Ia Ib Ic Ires                   (now)
  7 channels  same minus the window 25 cycles earlier  (superimposed / change)
Trained on the same train events and evaluated with the same trip logic and
test events as the feature models, so results are directly comparable.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ltline.features import (FIRST_CYCLE, SPC, W, estimate_frequency,  # noqa: E402
                             resample_cycles)
from ltline.protection import evaluate_events, summarise, tune  # noqa: E402
from ltline.sim.feeder import CLASSES, FeederSimulator  # noqa: E402

OUT = ROOT / "models" / "v2"
LAG = 25
EPOCHS = 14
DETECT_CYCLES = 35
torch.set_num_threads(2)
torch.manual_seed(0)


def event_tensor(kind, seed):
    ev = FeederSimulator().simulate(kind, int(seed))
    f0 = estimate_frequency(ev.signals[0], ev.fs)
    cyc = resample_cycles(ev.signals.astype(np.float64), ev.fs, f0)      # (6, ncyc, SPC)
    res = cyc[3:].sum(0, keepdims=True)
    x = np.concatenate([cyc[:3] / 325.0, cyc[3:] / 283.0, res / 283.0])  # (7, ncyc, SPC)
    return x, f0, ev


def window(x, c):
    now = x[:, c - W + 1: c + 1].reshape(7, -1)
    past = x[:, c - W + 1 - LAG: c + 1 - LAG].reshape(7, -1)
    delta = (now - past) * 10.0                                           # amplify small changes
    return np.clip(np.concatenate([now, delta]), -10, 10).astype(np.float32)


class Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, stride=2, padding=k // 2),
                                 nn.BatchNorm1d(o), nn.ReLU())
        self.body = nn.Sequential(blk(14, 32, 7), blk(32, 64, 5), blk(64, 96, 5),
                                  blk(96, 128, 3), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.body(x))


def sample_windows(man, per_event, rng):
    X, y = [], []
    for _, r in man.iterrows():
        x, f0, ev = event_tensor(r.kind, r.seed)
        ncyc = x.shape[1]
        cycles = np.arange(FIRST_CYCLE, ncyc)
        if r.label == "Normal":
            if pd.isna(r.onset):
                pick = [(c, "Normal") for c in rng.choice(cycles, per_event, replace=False)]
            else:
                # hard negatives: most windows right after the look-alike starts
                oc = r.onset * f0
                zone = cycles[(cycles + 1 > oc) & (cycles + 1 <= oc + DETECT_CYCLES)]
                rest = cycles[~np.isin(cycles, zone)]
                n_zone = min(len(zone), int(per_event * 0.7))
                pick = [(c, "Normal") for c in rng.choice(zone, n_zone, replace=False)]
                pick += [(c, "Normal") for c in rng.choice(rest, per_event - n_zone, replace=False)]
        else:
            oc = r.onset * f0
            pre = cycles[cycles + 1 < oc]
            zone = cycles[(cycles + 1 > oc) & (cycles + 1 <= oc + DETECT_CYCLES)]
            n_pre = min(len(pre), per_event // 3)
            pick = [(c, "Normal") for c in rng.choice(pre, n_pre, replace=False)] if n_pre else []
            pick += [(c, r.label) for c in rng.choice(zone, min(len(zone), per_event - n_pre), replace=False)]
        for c, lab in pick:
            X.append(window(x, c))
            y.append(CLASSES.index(lab))
    return torch.tensor(np.stack(X)), torch.tensor(y)


def stream_proba(net, man):
    """Probabilities for every window of every event, in the same layout as the features table."""
    rows, probs = [], []
    net.eval()
    with torch.no_grad():
        for _, r in man.iterrows():
            x, f0, ev = event_tensor(r.kind, r.seed)
            cs = np.arange(FIRST_CYCLE, x.shape[1])
            xb = torch.tensor(np.stack([window(x, c) for c in cs]))
            probs.append(torch.softmax(net(xb), 1).numpy())
            t = (cs + 1) / f0
            rows.append(pd.DataFrame({"event_id": r.event_id, "kind": r.kind, "label": r.label,
                                      "cycle": cs, "t": t,
                                      "since_onset": np.nan if pd.isna(r.onset) else t - r.onset}))
    return pd.concat(rows, ignore_index=True), np.concatenate(probs)


def main():
    rng = np.random.default_rng(0)
    man = pd.read_csv(ROOT / "data" / "sim_manifest.csv")
    train = man[man.split == "train"]
    val_ids = set(rng.choice(train.event_id, size=int(0.2 * len(train)), replace=False))
    fit, val, test = train[~train.event_id.isin(val_ids)], train[train.event_id.isin(val_ids)], man[man.split == "test"]

    X, y = sample_windows(fit, 20, rng)
    print("training windows", tuple(X.shape))
    counts = torch.bincount(y, minlength=len(CLASSES)).float()
    weights = (counts.sum() / (len(CLASSES) * counts.clamp(min=1)))
    net = Net(len(CLASSES))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * (len(X) // 256 + 1))
    lossf = nn.CrossEntropyLoss(weight=weights)
    for epoch in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            xb = X[b]
            xb = xb * (1 + 0.05 * torch.randn(len(b), 1, 1))       # amplitude augmentation
            opt.zero_grad()
            loss = lossf(net(xb), y[b])
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        print(f"epoch {epoch + 1}: loss {tot / len(X):.4f}", flush=True)

    # tune operating point on validation events, evaluate once on test
    win_val, p_val = stream_proba(net, val)
    th, k, val_s = tune(win_val, p_val, CLASSES)
    win_te, p_te = stream_proba(net, test)
    ev = evaluate_events(win_te, p_te, CLASSES, th, k)
    s = summarise(ev)
    print(f"cnn th={th} K={k}  dependability {s['dependability']:.3f}  correct {s['correct_type']:.3f}  "
          f"broken {s['broken_conductor_caught']:.3f}  false trips {s['false_trips_normal']}+"
          f"{s['false_trips_early']}  delay {s['median_delay_ms']:.0f} ms")
    ev.assign(model="cnn").to_csv(OUT / "events_test_cnn.csv", index=False)
    torch.save({"state": net.state_dict(), "threshold": th, "k": k}, OUT / "cnn.pt")
    res_path = OUT / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    results["cnn"] = {"summary": s, "threshold": th, "k_cycles": k, "val": val_s}
    res_path.write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
