"""LTLine Smart Protection System: Streamlit dashboard.

    streamlit run app.py

Streams current/voltage readings from an LT feeder (healthy, then a fault),
runs a 3-layer pipeline on every sample, and trips the line when a fault is confirmed:
  1. Detect   - Isolation Forest anomaly score -> early warning
  2. Protect  - Random Forest classifies the fault; persistent + confident -> isolator trips
  3. Respond  - simulated SCADA relay, resident SMS alert, crew dispatch
"""
from __future__ import annotations

import random
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ltline.core import (AGREE_TO_CONFIRM, CLASSES, FAULTS, FEATURES, PERSIST_TO_CONFIRM, SAFE_CURRENT_MAX, SAFE_VOLT_MAX,
                         SAFE_VOLT_MIN, WINDOW, load_models, load_segments,
                         make_features, models_exist)

st.set_page_config(page_title="LTLine · Smart Protection", page_icon="⚡", layout="wide")



# ---------- artifacts ----------
@st.cache_resource(show_spinner="Training models (first run only)…")
def get_models():
    if not models_exist():
        import train
        train.main()
    return load_models()


@st.cache_data
def get_segments():
    return load_segments()


iso, rf, meta = get_models()
segments = get_segments()
rf_classes = list(rf.classes_)


# ---------- scenario ----------
def contiguous(seg: pd.DataFrame, n: int) -> pd.DataFrame:
    n = min(n, len(seg))
    start = random.randint(0, len(seg) - n)
    return seg.iloc[start:start + n]


def new_scenario(fault_choice: str) -> pd.DataFrame:
    fault = random.choice(FAULTS) if fault_choice == "Random" else fault_choice
    healthy = contiguous(segments["Healthy"], random.randint(60, 140))
    faulty = contiguous(segments[fault], 60)
    return pd.concat([healthy, faulty], ignore_index=True)


def reset_state(fault_choice: str = "Random"):
    st.session_state.update(
        stream=new_scenario(fault_choice), idx=0, running=False, line="ON",
        phase="healthy", persist=0, recent_preds=[], logs=[], history=[], trip_info=None,
    )
    log("🟢 Monitoring started. Feeder energised, isolator closed.")


def log(msg: str):
    st.session_state.logs.append(f"{datetime.now():%H:%M:%S}  {msg}")


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Simulation")
    fault_choice = st.selectbox("Fault to inject", ["Random"] + FAULTS)
    speed_ms = st.slider("Speed (ms per sample)", 50, 800, 200, 50)
    anomaly_thresh = st.number_input(
        "Anomaly threshold", value=round(meta["default_anomaly_threshold"], 3),
        step=0.01, format="%.3f", help="Isolation Forest score above this raises an early warning. "
        "Default = 99th percentile of healthy training scores.")
    conf_thresh = st.slider("Classifier confidence to trip", 0.50, 0.99, 0.80, 0.01)
    window_size = st.slider("Chart window (samples)", 50, 300, 150, 25)
    st.divider()
    st.caption("Team Data Mavericks · Smart India Hackathon 2025 · PS #78 "
               "(software alternative to a circuit breaker for LT line breaks)")

if "stream" not in st.session_state:
    reset_state(fault_choice)


# ---------- pipeline step ----------
def step():
    s = st.session_state
    if s.idx >= len(s.stream) or s.line == "OFF":
        s.running = False
        return
    window = s.stream.iloc[max(0, s.idx - WINDOW + 1): s.idx + 1]
    x = make_features(window).iloc[[-1]][FEATURES]
    score = float(-iso.decision_function(x)[0])
    proba = rf.predict_proba(x)[0]
    k = int(np.argmax(proba))
    pred, conf = rf_classes[k], float(proba[k])
    row = s.stream.iloc[s.idx]

    warmed_up = s.idx >= WINDOW - 1   # rolling window full
    anomalous = warmed_up and score > anomaly_thresh
    s.persist = s.persist + 1 if anomalous else 0
    s.recent_preds = (s.recent_preds + [pred])[-AGREE_TO_CONFIRM:]

    if anomalous and s.phase == "healthy":
        s.phase = "warning"
        log(f"⚠️ Early warning at sample {s.idx}: anomaly score {score:.3f} > {anomaly_thresh:.3f}.")
        log("📡 Auto-check requested from SCADA. 👷 Crew on standby.")
    elif not anomalous and s.phase == "warning" and s.persist == 0:
        s.phase = "healthy"
        log("✅ Anomaly cleared. Transient only, no trip.")

    if (s.phase == "warning" and s.persist >= PERSIST_TO_CONFIRM and pred != "Healthy"
            and conf >= conf_thresh and len(set(s.recent_preds)) == 1
            and len(s.recent_preds) == AGREE_TO_CONFIRM):
        s.line, s.phase, s.running = "OFF", "tripped", False
        s.trip_info = dict(fault=pred, conf=conf, sample=s.idx,
                           current=float(row.current), voltage=float(row.voltage))
        log(f"❌ FAULT CONFIRMED: {pred} (confidence {conf:.0%}). Isolator OPEN, line de-energised.")
        log("📡 Trip relayed to control centre (SCADA).")
        log("📢 SMS/WhatsApp alert sent to residents near the affected pole.")
        log("👷 Nearest crew dispatched with GPS coordinates.")
        log(fault_detail(pred, row))

    s.history.append(dict(sample=s.idx, current=float(row.current), voltage=float(row.voltage),
                          true_label=row.label, predicted=pred, confidence=conf,
                          anomaly_score=score))
    s.idx += 1


def fault_detail(fault: str, row) -> str:
    return {
        "Overload": f"⚡ Current {row.current:.1f} A above rated {SAFE_CURRENT_MAX} A, sustained.",
        "Short Circuit": f"⚡ Current spike {row.current:.1f} A with voltage collapse to {row.voltage:.0f} V.",
        "Break": f"⚡ Current {row.current:.2f} A, voltage {row.voltage:.0f} V: conductor open, possible live wire on ground.",
        "Arc": "⚡ Erratic, jagged current/voltage: intermittent arcing at a loose joint.",
    }.get(fault, "")


# ---------- UI ----------
st.title("⚡ LTLine Smart Protection System")
st.caption("AI fault detection and automatic isolation for low-tension (230 V) distribution lines")

tab_live, tab_model, tab_how = st.tabs(["Live monitor", "Model performance", "How it works"])

with tab_live:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    if c1.button("▶️ Start", width="stretch", disabled=st.session_state.line == "OFF"):
        st.session_state.running = True
    if c2.button("⏸️ Pause", width="stretch"):
        st.session_state.running = False
    if c3.button("🔁 New scenario", width="stretch",
                 help="Restore the line and stream a fresh healthy → fault sequence"):
        reset_state(fault_choice)
        st.rerun()

    @st.fragment(run_every=speed_ms / 1000)
    def live():
        s = st.session_state
        if s.running:
            step()
            if not s.running:      # just tripped or ran out of data
                st.rerun()         # refresh the buttons outside the fragment

        hist = pd.DataFrame(s.history)
        last = hist.iloc[-1] if len(hist) else None

        colours = {"healthy": "#1b7f3a", "warning": "#c77700", "tripped": "#b42318"}
        if s.phase == "tripped":
            t = s.trip_info
            headline = f"❌ FAULT CONFIRMED: {t['fault']} · line isolated"
        elif s.phase == "warning":
            headline = f"⚠️ EARLY WARNING · anomaly score {last.anomaly_score:.3f}"
        else:
            headline = "🟢 HEALTHY"
        st.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{colours[s.phase]};color:white;"
            f"font-size:22px;font-weight:700;text-align:center'>{headline}<br>"
            f"<span style='font-size:16px;font-weight:500'>Line: {s.line} · Isolator: "
            f"{'OPEN' if s.line == 'OFF' else 'CLOSED'} · sample {s.idx}/{len(s.stream)}</span></div>",
            unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        if last is not None:
            m1.metric("Current", f"{last.current:.2f} A")
            m2.metric("Voltage", f"{last.voltage:.1f} V")
            m3.metric("Predicted", last.predicted, f"{last.confidence:.0%} confidence", delta_color="off", delta_arrow="off")
            m4.metric("Anomaly score", f"{last.anomaly_score:.3f}",
                      f"threshold {anomaly_thresh:.3f}", delta_color="off", delta_arrow="off")
        else:
            st.info("Press **Start** to begin streaming readings from the feeder.")

        if len(hist):
            view = hist.tail(window_size)
            cc, vc = st.columns(2)
            cc.altair_chart(signal_chart(view, "current", "Current (A)",
                                         None, SAFE_CURRENT_MAX), width="stretch")
            vc.altair_chart(signal_chart(view, "voltage", "Voltage (V)",
                                         SAFE_VOLT_MIN, SAFE_VOLT_MAX), width="stretch")

        lc, tc = st.columns([3, 2])
        with lc:
            st.markdown("**Event log**")
            st.code("\n".join(s.logs[-12:]) or "—", language=None, wrap_lines=True)
        with tc:
            st.markdown("**Latest readings**")
            if len(hist):
                st.dataframe(hist.tail(8)[["sample", "current", "voltage", "true_label",
                                           "predicted", "confidence", "anomaly_score"]]
                             .iloc[::-1].round(3), hide_index=True, width="stretch")

    def signal_chart(view, col, title, lo, hi):
        base = alt.Chart(view).mark_line(strokeWidth=2).encode(
            x=alt.X("sample:Q", title="Sample"),
            y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)))
        rules = [alt.Chart(pd.DataFrame({"y": [v]})).mark_rule(strokeDash=[5, 4], color="#b42318")
                 .encode(y="y:Q") for v in (lo, hi) if v is not None]
        return alt.layer(base, *rules).properties(height=230, title=f"{title} · dashed = safe limit")

    live()

with tab_model:
    st.subheader("How well the models work")
    a, b, c, d = st.columns(4)
    a.metric("Classifier accuracy", f"{meta['test_accuracy']:.1%}")
    b.metric("Macro F1", f"{meta['macro_f1']:.3f}")
    c.metric("Faults flagged early", f"{meta['anomaly_recall_on_faults']:.1%}",
             help="Share of fault samples whose Isolation Forest score crossed the threshold")
    d.metric("False-alarm rate", f"{meta['anomaly_false_alarm_rate']:.1%}",
             help="Share of healthy samples wrongly flagged")
    sc = meta["scenario_eval"]
    e, f, g = st.columns(3)
    e.metric("Correct trips (end-to-end)", f"{sc['correct_trip_rate']:.1%}",
             help="Full pipeline on held-out healthy → fault streams")
    f.metric("False trips", f"{sc['false_trips']} / {sc['scenarios']}")
    g.metric("Median time to trip", f"{sc['median_trip_delay_samples']:.0f} samples",
             help="Samples between fault onset and the isolator opening")
    st.caption(f"Test data: last 20% of each class segment in time order plus fault-onset windows "
               f"built from it ({meta['test_rows']} samples), never seen in training. "
               f"Scenario test: {sc['scenarios']} simulated streams from the same held-out data.")
    cm = pd.read_csv("models/confusion_matrix.csv", index_col=0)
    cm_long = cm.reset_index().melt(id_vars="index", var_name="Predicted", value_name="n") \
        .rename(columns={"index": "Actual"})
    heat = alt.Chart(cm_long).mark_rect().encode(
        x=alt.X("Predicted:N", sort=CLASSES), y=alt.Y("Actual:N", sort=CLASSES),
        color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
    text = alt.Chart(cm_long).mark_text(fontSize=14).encode(
        x=alt.X("Predicted:N", sort=CLASSES), y=alt.Y("Actual:N", sort=CLASSES), text="n:Q",
        color=alt.condition("datum.n > 150", alt.value("white"), alt.value("black")))
    st.altair_chart((heat + text).properties(height=320, title="Confusion matrix"),
                    width="stretch")
    st.markdown(
        "**What changed from the hackathon build (57% → 97.5% accuracy):**\n"
        "1. **Label fix.** The original dataset labelled 1,000 untouched healthy rows as *Break* "
        "and 1,700 as *Arc*, so the classifier was asked to tell identical readings apart. "
        "Removing them fixed the labels.\n"
        "2. **Window features.** Spread and sample-to-sample jumps over the last 5 readings let "
        "the model catch arcing, which shows up as a jagged waveform rather than one bad reading.\n"
        "3. **Fault-onset training.** Windows that straddle the healthy → fault boundary were added "
        "so that the step change at onset isn't mistaken for arcing. This cut Overload→Arc "
        "mis-trips from ~1 in 4 to ~3 in 100.\n"
        "4. **Trip confirmation** now needs 3 matching confident predictions, which removed false trips.")

with tab_how:
    st.subheader("Three layers")
    st.markdown(f"""
1. **Detect (early warning).** An Isolation Forest trained only on healthy readings scores every
   sample. A score above the threshold raises a warning *before* any fault is confirmed, and needs no fault labels.
2. **Protect (trip).** A Random Forest classifies the sample as {', '.join(CLASSES)}. The isolator
   trips only if the anomaly persists for {PERSIST_TO_CONFIRM}+ samples **and** the classifier gives the
   same fault {AGREE_TO_CONFIRM} times in a row with confidence ≥ the slider value. Short transients don't
   cause nuisance trips.
3. **Respond.** On a trip: SCADA relay, SMS/WhatsApp to residents near the pole, crew dispatch with GPS.
   This is simulated here; the deployed design uses Twilio and an edge relay.

**Features per sample:** current, voltage, plus rolling std and mean absolute change over the
last {WINDOW} samples for each signal.

**Fault signatures in the data**

| Fault | What happens |
|---|---|
| Break | Current → ~0 A, voltage collapses (open conductor) |
| Short Circuit | Current ~2.5× normal, voltage drops to ~20% |
| Overload | Current ~1.35× normal, voltage normal |
| Arc | Erratic, jagged current and voltage |

*Data is synthetic: 10,000 time-ordered readings across 5 classes, generated for the hackathon.
Real deployment would retrain on utility CT/PT logs.*
""")
