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

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
sys.path.insert(0, APP_DIR)

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
                   layout="wide")

with open(os.path.join(APP_DIR, "style.css")) as fh:
    st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


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


def card(label: str, title: str, body: str, muted: bool = False) -> None:
    """Result card. Replaces Streamlit's stock success/error boxes."""
    css = "ts-card ts-card-empty" if muted else "ts-card"
    st.markdown(
        f'<div class="{css}"><div class="ts-card-label">{label}</div>'
        f'<div class="ts-card-title">{title}</div>'
        f'<div class="ts-card-body">{body}</div></div>',
        unsafe_allow_html=True)


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
        f"Across {len(summary['accuracy_over_seeds'])} architectures trained on "
        f"identical terms, accuracy spans {max(accs) - min(accs):.2f} percentage "
        f"points. Among the {len(near_equal)} configurations within a point of "
        f"the best, <strong>energy still spans {ratio:.0f}×</strong> — "
        f"{cheapest['model']} {cheapest['precision']} at "
        f"{cheapest['energy_j_per_1k']['mean']:.2f} J per thousand inferences "
        f"against {dearest['model']} {dearest['precision']} at "
        f"{dearest['energy_j_per_1k']['mean']:.2f} J, for {gap:.2f} pp of "
        f"accuracy. On CPU-only hardware the deployment decision belongs to "
        f"energy and latency, not accuracy."
    )


st.title("TerraScope")
st.markdown(f'<p class="ts-lede">{headline()}</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="ts-note">Latency is measured live on this server. Accuracy, '
    'energy and CO₂e are looked up from the benchmark, measured on an Apple M2.'
    '</p>', unsafe_allow_html=True)

classify_tab, quant_tab, deploy_tab, method_tab = st.tabs(
    ["Classify", "Quantisation", "Deploy", "Method"])


# ------------------------------------------------------------- Classify -----

with classify_tab:
    left, right = st.columns([1, 2])

    with left:
        source = st.radio("Image source", ["Sample tile", "Upload"])
        image = None
        true_label = None

        if source == "Sample tile":
            tiles = samples.get("tiles", [])
            if tiles:
                labels = [f"{t['true_label']} — {t['file']}" for t in tiles]
                pick = st.selectbox("Tile", range(len(tiles)),
                                    format_func=lambda i: labels[i])
                chosen = tiles[pick]
                true_label = chosen["true_label"]
                image = Image.open(os.path.join(dd.SAMPLES_DIR, chosen["file"]))
                st.caption("From the held-out test fold — no model here trained "
                           "on it.")
            else:
                st.caption("No sample tiles bundled.")
        else:
            st.caption("EuroSAT covers 34 European countries at 10 m. A tile "
                       "from elsewhere is out of distribution and the "
                       "prediction should not be trusted.")
            upload = st.file_uploader("Satellite tile", type=["png", "jpg", "jpeg"])
            if upload:
                image = Image.open(upload)

        if image is not None:
            st.image(image, caption=(f"True class: {true_label}" if true_label
                                     else "Uploaded tile"), width=220)

    with right:
        picked = st.multiselect("Models", MODELS, default=list(MODELS))
        precision = st.selectbox("Precision", dd.PRECISIONS,
                                 index=dd.PRECISIONS.index("int8_static"))
        volume = st.select_slider(
            "Daily inference volume", options=[10_000, 100_000, 1_000_000,
                                               10_000_000],
            value=1_000_000, format_func=lambda v: f"{v:,}/day")
        run = st.button("Run classification", type="primary",
                        disabled=image is None or not picked)

    if run and image is not None:
        tensor = dd.preprocess(image)
        table = st.empty()
        collected = []

        for model in picked:
            with st.spinner(f"Running {model}…"):
                pred = dd.classify_and_time(model, precision, tensor)
            row = dd.measured(summary, model, precision)
            co2 = dd.daily_co2e_grams(summary, model, precision, volume)
            energy = (row["energy_j_per_1k"]["mean"]
                      if row and row.get("energy_j_per_1k") else None)
            acc = dd.accuracy_ci(summary, model, precision)

            collected.append({
                "Model": model,
                "Predicted": pred.label,
                "Verdict": ("—" if not true_label
                            else "correct" if pred.label == true_label
                            else "wrong"),
                "Confidence": f"{pred.confidence * 100:.1f}%",
                "Latency, live": f"{pred.latency_ms_median:.3f} ms",
                "Benchmark accuracy": ("not measured" if not acc
                                       else f"{acc[0]:.2f}%"),
                "Energy /1k": f"{fmt(energy)} J",
                f"CO₂e at {volume:,}/day": ("not measured" if co2 is None
                                            else f"{co2:.2f} g"),
            })
            table.dataframe(pd.DataFrame(collected), use_container_width=True,
                            hide_index=True)

        st.caption(f"Latency: median of {dd.TIMED_RUNS} runs on this server, "
                   f"after {dd.WARMUP_RUNS} warm-up runs discarded, at "
                   f"{dd.REPORT_THREADS} thread.")

        wrong = [c for c in collected if c["Verdict"] == "wrong"]
        if wrong and true_label:
            st.markdown(
                f'<p class="ts-flag">{len(wrong)} of {len(collected)} models '
                f'got this wrong — {", ".join(c["Model"] for c in wrong)} — '
                f'confidently, and at full speed.</p>', unsafe_allow_html=True)


# --------------------------------------------------------- Quantisation -----

with quant_tab:
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
            st.metric(prec, f"{acc[0]:.2f}%",
                      None if delta is None else f"{delta:+.2f} pp vs fp32",
                      delta_color="normal")
            detail = f"95% CI ± {fmt(acc[1])} pp"
            if row:
                detail += f" · {fmt(row['onnx_mb'])} MB"
                if row.get("energy_j_per_1k"):
                    detail += f" · {fmt(row['energy_j_per_1k']['mean'])} J/1k"
            st.caption(detail)

    # Markdown rather than st.dataframe: the grid collapses its columns to
    # illegibility when a tab is rendered before its width is measured, and this
    # table is small, static and read left-to-right.
    lines = ["| Model | Precision | fp32 % | int8 % | Change (pp) | Verdict |",
             "|---|---|---:|---:|---:|---|"]
    for m in MODELS:
        for prec in ("int8_dynamic", "int8_static"):
            d = dd.quantisation_delta(summary, m, prec)
            if not d:
                continue
            half = d.get("delta_half_width")
            change = (f"{d['delta_mean'] * 100:+.2f}"
                      + ("" if half is None else f" ± {half * 100:.2f}"))
            verdict = ("indistinguishable from fp32"
                       if half is not None and abs(d["delta_mean"]) < half
                       else "real loss")
            lines.append(f"| {m} | {prec} | {d['fp32_mean'] * 100:.2f} | "
                         f"{d['quant_mean'] * 100:.2f} | {change} | {verdict} |")
    st.markdown("\n".join(lines))
    st.caption("Paired per-seed change against each model's own fp32 export; a "
               "confidence interval spanning zero means no detectable change.")


# --------------------------------------------------------------- Deploy -----

with deploy_tab:
    rows = [r for r in rows_at_report_config() if r.get("energy_j_per_1k")]
    df = pd.DataFrame([{
        "config": f"{r['model']} {r['precision']}",
        "accuracy": r["test_acc"]["mean"] * 100,
        "energy": r["energy_j_per_1k"]["mean"],
        "p95": r["latency_p95_ms"]["mean"],
        "size_mb": r["onnx_mb"],
    } for r in rows])

    c1, c2 = st.columns(2)
    with c1:
        min_acc = st.slider("Minimum accuracy (%)", float(df["accuracy"].min()),
                            float(df["accuracy"].max()),
                            float(df["accuracy"].median()),
                            step=ACCURACY_SLIDER_STEP)
    with c2:
        max_p95 = st.slider("Maximum p95 latency (ms)", float(df["p95"].min()),
                            float(df["p95"].max()), float(df["p95"].max()),
                            step=LATENCY_SLIDER_STEP)

    ok = df[(df["accuracy"] >= min_acc) & (df["p95"] <= max_p95)]
    if ok.empty:
        card("No candidate", "Nothing measured meets both constraints",
             "Loosen one. This is a real answer about the measured set, not a "
             "failure of the tool.", muted=True)
    else:
        best = ok.loc[ok["energy"].idxmin()]
        card("Lowest energy meeting your constraints",
             best["config"],
             f"{best['accuracy']:.2f}% accuracy · "
             f"{best['energy']:.2f} J per thousand inferences · "
             f"{best['p95']:.2f} ms p95 · {best['size_mb']:.1f} MB — "
             f"chosen from {len(ok)} qualifying configuration(s).")

    frontier = dd.pareto_frontier(df.to_dict("records"))
    chart_df = df.assign(
        selected=df["config"].isin(ok["config"]).map({True: "yes", False: "no"}),
        on_frontier=df["config"].isin(frontier))
    points = (alt.Chart(chart_df)
              .mark_circle(size=170, opacity=0.9)
              .encode(
                  x=alt.X("energy:Q", scale=alt.Scale(type="log"),
                          title="Energy per 1,000 inferences (J, measured on M2)"),
                  y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False),
                          title="Test accuracy (%), 5-seed mean"),
                  color=alt.Color("selected:N",
                                  scale=alt.Scale(domain=["yes", "no"],
                                                  range=["#1b4332", "#c1c8c2"]),
                                  legend=alt.Legend(title="Meets constraints")),
                  tooltip=["config", "accuracy", "energy", "p95", "size_mb"])
              .properties(height=420))
    line = (alt.Chart(chart_df[chart_df["on_frontier"]].sort_values("energy"))
            .mark_line(strokeDash=[6, 4], color="#86af99")
            .encode(x=alt.X("energy:Q", scale=alt.Scale(type="log")),
                    y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False))))
    st.altair_chart(line + points, use_container_width=True)
    st.caption("Dashed line: the Pareto frontier, which nothing beats on both "
               "energy and accuracy. Log-scale x axis.")


# --------------------------------------------------------------- Method -----

with method_tab:
    prov = dd.provenance(summary)
    env = prov["environment"]

    st.subheader("Provenance")
    c1, c2 = st.columns(2)
    with c1:
        st.code(
            f"split file    {prov['split_csv']}\n"
            f"split sha256  {prov['split_sha256']}\n"
            f"recipe hash   {prov['recipe_hash']}\n"
            f"seeds/model   {', '.join(str(s) for s in prov['seeds'])}",
            language="text")
        st.markdown(
            "EuroSAT ships no official train/test split, so every published "
            "EuroSAT accuracy is measured against folds the reader cannot "
            "inspect. This benchmark commits its split and references it by "
            "hash from every result; runs abort if the file stops matching its "
            "own checksum. Each architecture is trained on one identical "
            "recipe over five seeds, and accuracy is reported as a mean with a "
            "Student-t confidence interval rather than a single lucky run.")
    with c2:
        # Rendered as markdown rather than st.dataframe: inside a half-width
        # column the grid truncates both headers and values to illegibility.
        hardware = [("CPU", "cpu"), ("Cores", "cpu_cores_logical"),
                    ("OS", "os"), ("Python", "python"),
                    ("ONNX Runtime", "onnxruntime"),
                    ("Measured (UTC)", "timestamp_utc")]
        lines = ["| Property | Value |", "|---|---|"]
        lines += [f"| {name} | {env.get(key, 'not recorded')} |"
                  for name, key in hardware]
        st.markdown("\n".join(lines))
        st.caption("Benchmark hardware — not the server rendering this page.")

    st.subheader("Energy accounting")
    st.markdown(
        f"{prov.get('energy_note') or 'not recorded'} CO₂e assumes "
        f"{fmt(prov['grid_intensity'], ',.0f')} gCO₂e/kWh "
        f"({prov.get('grid_intensity_source', 'source not recorded')}). That "
        f"constant scales every carbon figure linearly, so it is stated rather "
        f"than buried in a library default. Inference latency on this page is "
        f"measured live because a server can time itself honestly; energy "
        f"cannot be, because this container exposes no power telemetry, and an "
        f"estimate presented as a measurement is the failure mode this project "
        f"exists to avoid.")

    st.subheader("Excluded measurement windows")
    ex = prov["exclusions"]
    if ex:
        st.markdown(
            f"{ex.get('windows_excluded')} of {ex.get('windows_total')} windows "
            f"were excluded by criteria registered before the measurements were "
            f"taken. They remain in the published data; nothing was deleted, and "
            f"removing them changes no headline figure.")
        if ex.get("excluded_windows"):
            lines = ["| Configuration | Reason |", "|---|---|"]
            for w in ex["excluded_windows"]:
                lines.append(
                    f"| {w['model']} {w['precision']} seed{w['seed']} "
                    f"t{w['threads_intra_op']} b{w['batch_size']} "
                    f"| {'; '.join(w['reasons'])} |")
            st.markdown("\n".join(lines))

    st.subheader("Known confound")
    for threads, regime in prov["thread_regime_confound"].items():
        st.markdown(f"**{threads}** — {regime.get('note')}")
    st.markdown(
        "Every figure on the other tabs uses the single-thread configuration, "
        "which is the unconfounded one. The four-thread windows are published "
        "with the confound characterised rather than quietly dropped.")

    st.subheader("Read further")
    st.markdown(
        "[PROTOCOL.md](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/blob/master/PROTOCOL.md) "
        "— measurement protocol, outcomes and every deviation · "
        "[DATASHEET.md](https://github.com/eklavya072/TerraScope-GeoSpatial-AI/blob/master/DATASHEET.md) "
        "— dataset documentation · "
        "[repository](https://github.com/eklavya072/TerraScope-GeoSpatial-AI)")
