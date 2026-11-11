| Approach | Faults tripped | Right type | Broken conductor caught | False trips | SC trip time |
|---|---|---|---|---|---|
| Conventional relay (OC + EF) | 38.2% | 17.5% | 0.0% | 1 / 240 | 41 ms |
| v1-style ML (RMS only) | 77.0% | 55.8% | 88.8% | 114 / 240 | 47 ms |
| Random Forest (relay features) | 96.0% | 83.2% | 90.6% | 1 / 240 | 30 ms |
| LightGBM (relay features) | 99.2% | 82.8% | 98.1% | 0 / 240 | 30 ms |
| 1D-CNN (raw waveforms) | 98.0% | 79.2% | 99.4% | 18 / 240 | 30 ms |

Faults tripped, per class:

| Approach | ShortCircuit | Overload | OpenConductor | DownedConductor | SeriesArc |
|---|---|---|---|---|---|
| Conventional relay (OC + EF) | 100.0% | 56.2% | 22.5% | 12.5% | 0.0% |
| v1-style ML (RMS only) | 97.5% | 93.8% | 90.0% | 90.0% | 13.8% |
| Random Forest (relay features) | 98.8% | 100.0% | 90.0% | 91.2% | 100.0% |
| LightGBM (relay features) | 100.0% | 100.0% | 98.8% | 97.5% | 100.0% |
| 1D-CNN (raw waveforms) | 92.5% | 100.0% | 100.0% | 98.8% | 98.8% |
