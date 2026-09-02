"""TerraScope demo — see the benchmark's finding in about a minute.

Rule this file obeys without exception: it never displays a number it did not
either measure live in this session or read from results/summary.json. There
are no numeric literals describing results anywhere below. Latency is measured
here, on this server. Accuracy, energy and CO2e are looked up from the committed
benchmark and labelled as having been measured on different hardware.

Run:  streamlit run app_demo.py
"""

from __future__ import annotations

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import demo_data as dd  # noqa: E402

# Structural constants for the UI. These are NOT measured results -- they are a
# tolerance and two slider increments -- and tests/test_demo_app.py allows
# decimal literals only on the lines that define them, so any other decimal in
# this file is a test failure. That is the mechanism keeping invented figures
# out of the app.
NEAR_EQUAL_ACCURACY_TOLERANCE = 0.01   # 1 percentage point, as a fraction
ACCURACY_SLIDER_STEP = 0.05            # percentage points
LATENCY_SLIDER_STEP = 0.05             # milliseconds

st.set_page_config(page_title="TerraScope — accuracy vs energy on CPU",
                   page_icon="🛰️", layout="wide")


# ----------------------------------------------------------------- loading ---

@st.cache_data(show_spinner=False)
def get_summary() -> dict:
    return dd.load_summary()


@st.cache_data(show_spinner=False)
def get_samples() -> dict:
    return dd.load_samples()


summary = get_summary()
samples = get_samples()
MODELS = dd.available_models(summary)


def fmt(value, spec: str = ",.2f", missing: str = "not measured") -> str:
    """One place where an unavailable number becomes honest text."""
    if value is None:
        return missing
    return format(value, spec)


def rows_at_report_config() -> list[dict]:
    return [v for v in summary["measured"].values()
            if v["threads"] == dd.REPORT_THREADS
            and v["batch_size"] == dd.REPORT_BATCH]


# ------------------------------------------------------------------ header ---

def headline() -> str:
    """The finding, computed from the artefacts rather than typed in.

    Framing: among configurations that are all within one percentage point of
    the best accuracy measured -- i.e. among options a deployer would consider
    equivalent on accuracy -- how far apart is their energy? That is the
    comparison the benchmark exists to make, and it avoids the trap of quoting a
    ratio between an accurate model and an unusable one.
    """
    rows = [r for r in rows_at_report_config() if r.get("energy_j_per_1k")]
    if not rows:
        return "No measured configurations are available."

    best = max(rows, key=lambda r: r["test_acc"]["mean"])
    near_equal = [r for r in rows
                  if r["test_acc"]["mean"]
                  >= best["test_acc"]["mean"] - NEAR_EQUAL_ACCURACY_TOLERANCE]
    cheapest = min(near_equal, key=lambda r: r["energy_j_per_1k"]["mean"])
    dearest = max(near_equal, key=lambda r: r["energy_j_per_1k"]["mean"])
    ratio = dearest["energy_j_per_1k"]["mean"] / cheapest["energy_j_per_1k"]["mean"]
    gap = (best["test_acc"]["mean"] - cheapest["test_acc"]["mean"]) * 100
    accs = [v["mean"] * 100 for v in summary["accuracy_over_seeds"].values()]

    return (
        f"Across {len(summary['accuracy_over_seeds'])} architectures, accuracy "
        f"spans **{max(accs) - min(accs):.2f} percentage points**. Among the "
        f"{len(near_equal)} configurations within 1 pp of the best accuracy "
        f"measured, **energy still spans {ratio:.0f}×** — "
        f"`{cheapest['model']} {cheapest['precision']}` at "
        f"**{cheapest['energy_j_per_1k']['mean']:.2f} J** per 1,000 inferences "
        f"against `{dearest['model']} {dearest['precision']}` at "
        f"**{dearest['energy_j_per_1k']['mean']:.2f} J**, for "
        f"**{gap:.2f} pp** of accuracy. On CPU-only hardware the deployment "
        f"decision belongs to energy and latency, not accuracy."
    )


st.title("🛰️ TerraScope")
st.markdown(headline())

st.warning(
    "**How to read every number here.** Latency is **measured live on this "
    "server**, right now, as you click. Accuracy, energy and CO₂e are **looked "
    "up from the committed benchmark**, measured on an Apple M2 — this container "
    "has no power telemetry, so energy is never computed here. Accuracy comes "
    "from a 5-seed run against a held-out split. Nothing on this page is an "
    "estimate: anything unmeasured says *not measured*.",
    icon="⚖️")

race_tab, quant_tab, pareto_tab, receipts_tab = st.tabs(
    ["① The race", "② Quantisation roulette", "③ Which should I deploy?",
     "④ Receipts"])


# --------------------------------------------------------- ① the race -------

with race_tab:
    st.subheader("Classify a tile on every model at once")
    st.caption("Every sample below is from the **held-out test fold** of the "
               "committed split — no model here was trained on any of them.")

    left, right = st.columns([1, 2])

    with left:
        source = st.radio("Image source", ["Sample tile (test fold)", "Upload"],
                          horizontal=False)
        image = None
        true_label = None

        if source.startswith("Sample"):
            tiles = samples.get("tiles", [])
            if tiles:
                labels = [f"{t['true_label']} — {t['file']}" for t in tiles]
                pick = st.selectbox("Tile", range(len(tiles)),
                                    format_func=lambda i: labels[i])
                chosen = tiles[pick]
                true_label = chosen["true_label"]
                image = Image.open(os.path.join(dd.SAMPLES_DIR, chosen["file"]))
            else:
                st.info("No sample tiles bundled.")
        else:
            st.info("**Before you upload:** EuroSAT covers 34 European countries "
                    "at 10 m resolution. A tile from another continent, another "
                    "sensor, or a different scale is outside the distribution "
                    "these models were trained on, and the prediction should not "
                    "be trusted — the models will still return a confident "
                    "answer, which is exactly the problem.", icon="🌍")
            upload = st.file_uploader("Satellite tile", type=["png", "jpg", "jpeg"])
            if upload:
                image = Image.open(upload)

        if image is not None:
            st.image(image, caption=(f"True class: {true_label}" if true_label
                                     else "Uploaded tile"), width=220)
            if true_label:
                st.caption("The true class is known because this tile comes from "
                           "the labelled test fold.")

    with right:
        default = [m for m in MODELS]
        picked = st.multiselect(
            "Models to race (all int8-static by default — watch what "
            "quantisation does to some of them)", MODELS, default=default)
        precision = st.selectbox("Precision", dd.PRECISIONS,
                                 index=dd.PRECISIONS.index("int8_static"))
        volume = st.select_slider(
            "Daily inference volume (for the CO₂e column)",
            options=[10_000, 100_000, 1_000_000, 10_000_000],
            value=1_000_000,
            format_func=lambda v: f"{v:,}/day")

        run = st.button("Run classification", type="primary",
                        disabled=image is None or not picked)

    if run and image is not None:
        tensor = dd.preprocess(image)
        table = st.empty()
        progress = st.progress(0.0)
        collected = []

        for i, model in enumerate(picked, start=1):
            with st.spinner(f"Running {model} ({precision})…"):
                pred = dd.classify_and_time(model, precision, tensor)
            row = dd.measured(summary, model, precision)
            co2 = dd.daily_co2e_grams(summary, model, precision, volume)
            energy = (row["energy_j_per_1k"]["mean"]
                      if row and row.get("energy_j_per_1k") else None)
            acc = dd.accuracy_ci(summary, model, precision)

            collected.append({
                "Model": model,
                "Predicted": pred.label,
                "Correct": ("—" if not true_label
                            else "✅" if pred.label == true_label else "❌"),
                "Confidence": f"{pred.confidence * 100:.1f}%",
                "Latency (live, median)": f"{pred.latency_ms_median:.3f} ms",
                "Benchmark accuracy": ("not measured" if not acc
                                       else f"{acc[0]:.2f}%"),
                "Energy /1k (M2)": f"{fmt(energy)} J",
                f"CO₂e at {volume:,}/day": ("not measured" if co2 is None
                                            else f"{co2:.2f} g"),
            })
            table.dataframe(pd.DataFrame(collected), use_container_width=True,
                            hide_index=True)
            progress.progress(i / len(picked))

        progress.empty()
        st.caption(f"Latency is the **median of {dd.TIMED_RUNS} runs measured on "
                   f"this server** after {dd.WARMUP_RUNS} discarded warm-up runs, "
                   f"at {dd.REPORT_THREADS} thread. It reflects this container's "
                   f"CPU, not the benchmark hardware. Energy, CO₂e and accuracy "
                   f"are looked up from the committed benchmark (Apple M2).")

        wrong = [c for c in collected if c["Correct"] == "❌"]
        if wrong and true_label:
            st.error(
                f"**{len(wrong)} of {len(collected)} models got this wrong** — "
                f"{', '.join(c['Model'] for c in wrong)}. Confidently, and at "
                f"full speed. Speed and confidence are not accuracy; see the "
                f"next tab for what quantisation did to them.", icon="⚠️")


# ------------------------------------------- ② quantisation roulette --------

with quant_tab:
    st.subheader("What post-training quantisation costs")
    st.caption("Shrinking a model to int8 is close to free for some "
               "architectures and destroys others. All figures below are from "
               "the committed 5-seed benchmark.")

    model = st.selectbox("Architecture", MODELS, key="quant_model")
    cols = st.columns(len(dd.PRECISIONS))

    base = dd.accuracy_ci(summary, model, "fp32")
    for col, prec in zip(cols, dd.PRECISIONS):
        acc = dd.accuracy_ci(summary, model, prec)
        row = dd.measured(summary, model, prec)
        with col:
            if acc is None:
                st.metric(prec, "not measured")
                continue
            delta = None if prec == "fp32" or base is None else acc[0] - base[0]
            st.metric(
                prec,
                f"{acc[0]:.2f}%",
                None if delta is None else f"{delta:+.2f} pp vs fp32",
                delta_color="normal")
            st.caption(f"95% CI ± {fmt(acc[1])} pp")
            if row:
                st.caption(f"{fmt(row['onnx_mb'])} MB on disk")
                if row.get("energy_j_per_1k"):
                    st.caption(f"{fmt(row['energy_j_per_1k']['mean'])} J / 1k")

    st.divider()
    st.markdown("**Paired per-seed accuracy change against each model's own "
                "fp32 export.** A confidence interval containing zero means the "
                "change is not distinguishable from no change at all.")

    qrows = []
    for m in MODELS:
        for prec in ("int8_dynamic", "int8_static"):
            d = dd.quantisation_delta(summary, m, prec)
            if not d:
                continue
            half = d.get("delta_half_width")
            qrows.append({
                "Model": m,
                "Precision": prec,
                "fp32 %": f"{d['fp32_mean'] * 100:.2f}",
                "int8 %": f"{d['quant_mean'] * 100:.2f}",
                "Δ (pp)": f"{d['delta_mean'] * 100:+.2f}"
                          + ("" if half is None else f" ± {half * 100:.2f}"),
                "Verdict": ("indistinguishable from fp32"
                            if half is not None
                            and abs(d["delta_mean"]) < half else "real loss"),
            })
    st.dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)
    st.caption("int8-dynamic is worse than fp32 on **both** accuracy and energy "
               "for every model measured — it recomputes activation ranges on "
               "every call. It is reported because a negative result saves "
               "someone else the experiment.")


# ------------------------------------------------- ③ the Pareto picker ------

with pareto_tab:
    st.subheader("Which configuration should I actually deploy?")

    rows = [r for r in rows_at_report_config() if r.get("energy_j_per_1k")]
    df = pd.DataFrame([{
        "config": f"{r['model']} {r['precision']}",
        "model": r["model"],
        "precision": r["precision"],
        "accuracy": r["test_acc"]["mean"] * 100,
        "energy": r["energy_j_per_1k"]["mean"],
        "p95": r["latency_p95_ms"]["mean"],
        "size_mb": r["onnx_mb"],
    } for r in rows])

    c1, c2 = st.columns(2)
    with c1:
        min_acc = st.slider("Minimum accuracy (%)",
                            float(df["accuracy"].min()),
                            float(df["accuracy"].max()),
                            float(df["accuracy"].median()),
                            step=ACCURACY_SLIDER_STEP)
    with c2:
        max_p95 = st.slider("Maximum p95 latency (ms)",
                            float(df["p95"].min()), float(df["p95"].max()),
                            float(df["p95"].max()), step=LATENCY_SLIDER_STEP)

    ok = df[(df["accuracy"] >= min_acc) & (df["p95"] <= max_p95)]
    if ok.empty:
        st.error("No measured configuration satisfies both constraints. "
                 "Loosen one — this is a real answer, not a failure.")
    else:
        best = ok.loc[ok["energy"].idxmin()]
        st.success(
            f"**{best['config']}** — {best['accuracy']:.2f}% accuracy, "
            f"{best['energy']:.2f} J per 1,000 inferences, "
            f"{best['p95']:.2f} ms p95, {best['size_mb']:.1f} MB. "
            f"Lowest energy among {len(ok)} configuration(s) meeting your "
            f"constraints.", icon="✅")

    frontier = dd.pareto_frontier(df.to_dict("records"))
    chart_df = df.assign(selected=df["config"].isin(ok["config"]),
                         on_frontier=df["config"].isin(frontier))
    points = (alt.Chart(chart_df)
              .mark_circle(size=180, opacity=0.9)
              .encode(
                  x=alt.X("energy:Q", scale=alt.Scale(type="log"),
                          title="Energy per 1,000 inferences (J, measured on M2)"),
                  y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False),
                          title="Test accuracy (%), 5-seed mean"),
                  color=alt.Color("selected:N",
                                  scale=alt.Scale(domain=[True, False],
                                                  range=["#c1440e", "#b9b9b9"]),
                                  legend=alt.Legend(title="Meets constraints")),
                  tooltip=["config", "accuracy", "energy", "p95", "size_mb"])
              .properties(height=430))

    # The frontier is the set of configurations nothing else beats on BOTH
    # energy and accuracy. It includes points that are cheap but useless -- that
    # is what the frontier means, and hiding them would be editing the result.
    line = (alt.Chart(chart_df[chart_df["on_frontier"]].sort_values("energy"))
            .mark_line(strokeDash=[6, 4], color="#c1440e", opacity=0.8)
            .encode(x=alt.X("energy:Q", scale=alt.Scale(type="log")),
                    y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False))))
    st.altair_chart(line + points, use_container_width=True)
    st.caption("Log-scale x axis. The dashed line is the Pareto frontier: "
               "configurations nothing else beats on both energy and accuracy. "
               "It includes cheap-but-unusable points, because that is what the "
               "frontier means — the accuracy slider is how you exclude them. "
               "Energy and accuracy are measured figures from the benchmark; "
               "the filtering is the only thing computed here.")


# ------------------------------------------------------- ④ receipts ---------

with receipts_tab:
    st.subheader("Why you should believe the numbers on the other three tabs")
    prov = dd.provenance(summary)
    env = prov["environment"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Provenance**")
        st.code(
            f"split file    {prov['split_csv']}\n"
            f"split sha256  {prov['split_sha256']}\n"
            f"recipe hash   {prov['recipe_hash']}\n"
            f"seeds/model   {', '.join(str(s) for s in prov['seeds'])}",
            language="text")
        st.caption("Every accuracy figure references that split by hash. "
                   "EuroSAT ships no official train/test split, so the fold "
                   "definition is committed to the repository and verified "
                   "against its own checksum on every run.")
    with c2:
        st.markdown("**Benchmark hardware** (not this server)")
        st.dataframe(pd.DataFrame(
            [{"Property": k, "Value": str(env.get(v, "not recorded"))}
             for k, v in [("CPU", "cpu"), ("Cores", "cpu_cores_logical"),
                          ("OS", "os"), ("Python", "python"),
                          ("ONNX Runtime", "onnxruntime"),
                          ("Measured (UTC)", "timestamp_utc")]]),
            use_container_width=True, hide_index=True)

    st.markdown("**Energy accounting**")
    st.caption(prov.get("energy_note") or "not recorded")
    st.caption(f"CO₂e assumes {fmt(prov['grid_intensity'], ',.0f')} gCO₂e/kWh — "
               f"{prov.get('grid_intensity_source', 'source not recorded')}.")

    ex = prov["exclusions"]
    st.markdown("**Excluded measurement windows**")
    if ex:
        st.caption(f"{ex.get('windows_excluded')} of {ex.get('windows_total')} "
                   f"windows were excluded by criteria registered *before* the "
                   f"measurements were taken. They remain in the published data; "
                   f"nothing was deleted.")
        if ex.get("excluded_windows"):
            st.dataframe(pd.DataFrame([{
                "Config": f"{w['model']} {w['precision']} seed{w['seed']} "
                          f"t{w['threads_intra_op']} b{w['batch_size']}",
                "Reason": "; ".join(w["reasons"]),
            } for w in ex["excluded_windows"]]),
                use_container_width=True, hide_index=True)

    st.markdown("**Known confound**")
    for threads, regime in prov["thread_regime_confound"].items():
        icon = "⚠️" if regime.get("bimodal") else "✅"
        st.caption(f"{icon} **{threads}** — {regime.get('note')}")
    st.caption("Everything on the other tabs uses the single-thread "
               "configuration, which is the unconfounded one.")

    st.markdown(
        "**Read further** — "
        "[PROTOCOL.md](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/blob/master/PROTOCOL.md) "
        "(measurement protocol, outcomes and every deviation) · "
        "[DATASHEET.md](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/blob/master/DATASHEET.md) · "
        "[repository](https://github.com/eklavya072/TerraScope-GeoSpatial-AI)")
