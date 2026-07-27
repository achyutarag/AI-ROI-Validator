"""
Main Streamlit app for the AI Bootstrap Auditor toolkit.

Structure (internal notes -- not shown to users):
  - Tab 1: Single Metric Audit   -- AIBootstrapAuditor, recomputes live, sub-second
  - Tab 2: Guardrail Check       -- GuardrailAuditor, recomputes live, sub-second
  - Tab 3: Batch Review          -- AuditBatch, recomputes live, sub-second
  - Tab 4: Sample Size Planner   -- cached + button-triggered, can take a while (see slow_path_cache.py)
  - Tab 5: Harness Calibration   -- cached + button-triggered, advanced/optional, can take a while

Run locally:
    pip install streamlit pandas numpy
    streamlit run audit_toolkit_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

from ai_bootstrap_auditor import AIBootstrapAuditor, AuditBatch, GuardrailAuditor
from slow_path_cache import sample_size_planner_section, harness_calibration_section

st.set_page_config(page_title="AI Bootstrap Auditor", layout="wide")


# =====================================================================
# SHARED DATA-INPUT HELPER (used by tabs 1-3)
# =====================================================================
GUARDRAIL_DIRECTION_OPTIONS = ["not_increase", "not_decrease"]
GUARDRAIL_DIRECTION_LABELS = {
    "not_increase": "go up (e.g. latency, cost, error rate)",
    "not_decrease": "go down (e.g. accuracy, relevance)",
}

EXAMPLE_SCENARIOS = {
    "(custom / paste your own)": None,
    "Real accuracy win (70% -> 78%, n=500)": {
        "seed": 42, "before_p": 0.70, "after_p": 0.78, "n": 500, "kind": "binomial",
    },
    "Borderline effect (70% -> 74%, n=80)": {
        "seed": 7, "before_p": 0.70, "after_p": 0.74, "n": 80, "kind": "binomial",
    },
    "No real effect (70% -> 70%, n=300)": {
        "seed": 11, "before_p": 0.70, "after_p": 0.70, "n": 300, "kind": "binomial",
    },
    "Latency improvement (1200ms -> 950ms, n=150)": {
        "seed": 42, "before_mean": 1200, "after_mean": 950, "scale": 275, "n": 150, "kind": "normal",
    },
    "Token count regression (180 -> 214, n=250)": {
        "seed": 11, "before_mean": 180, "after_mean": 214, "scale": 42, "n": 250, "kind": "normal",
    },
}


def _generate_example(spec: dict) -> tuple:
    rng = np.random.RandomState(spec["seed"])
    if spec["kind"] == "binomial":
        before = rng.binomial(1, p=spec["before_p"], size=spec["n"])
        after = rng.binomial(1, p=spec["after_p"], size=spec["n"])
    else:  # normal
        before = np.clip(rng.normal(loc=spec["before_mean"], scale=spec["scale"], size=spec["n"]), 1, None)
        after = np.clip(rng.normal(loc=spec["after_mean"], scale=spec["scale"], size=spec["n"]), 1, None)
    return before, after


def data_input_widget(key_prefix: str) -> tuple:
    """
    Returns (before_array, after_array) or (None, None) if not yet provided.
    Three input methods: example scenario, CSV upload (columns 'before','after'),
    or manual comma-separated paste. Shared across tabs 1-3 so behavior is
    consistent everywhere data is needed.
    """
    method = st.radio(
        "Data source", ["Example scenario", "Upload CSV", "Paste values"],
        horizontal=True, key=f"{key_prefix}_method",
    )

    if method == "Example scenario":
        choice = st.selectbox("Scenario", list(EXAMPLE_SCENARIOS.keys()), key=f"{key_prefix}_scenario")
        spec = EXAMPLE_SCENARIOS[choice]
        if spec is None:
            st.info("Select a scenario, or switch to Upload/Paste for real data.")
            return None, None
        return _generate_example(spec)

    elif method == "Upload CSV":
        st.caption("CSV must have two columns named exactly 'before' and 'after', same length, paired by row.")
        uploaded = st.file_uploader("Upload CSV", type="csv", key=f"{key_prefix}_upload")
        if uploaded is None:
            return None, None
        try:
            df = pd.read_csv(uploaded)
            return df["before"].to_numpy(), df["after"].to_numpy()
        except Exception as e:
            st.error(f"Couldn't read CSV: {e}")
            return None, None

    else:  # Paste values
        col1, col2 = st.columns(2)
        before_text = col1.text_area("Before (comma-separated)", key=f"{key_prefix}_before_text",
                                      placeholder="0.72, 0.68, 0.75, ...")
        after_text = col2.text_area("After (comma-separated)", key=f"{key_prefix}_after_text",
                                     placeholder="0.80, 0.77, 0.81, ...")
        if not before_text.strip() or not after_text.strip():
            return None, None
        try:
            before = np.array([float(x.strip()) for x in before_text.split(",") if x.strip()])
            after = np.array([float(x.strip()) for x in after_text.split(",") if x.strip()])
        except ValueError:
            st.error("Couldn't parse values -- check for stray characters.")
            return None, None
        if len(before) != len(after):
            st.error(f"Before ({len(before)} values) and after ({len(after)} values) must be paired -- same length.")
            return None, None
        return before, after


# =====================================================================
# TAB 1: SINGLE METRIC AUDIT (fast path)
# =====================================================================
def single_metric_audit_tab():
    st.subheader("Single Metric Audit")
    st.caption("Is this metric a real win, or could it be noise? Updates instantly as you change inputs.")

    metric_name = st.text_input("Metric name", value="My Metric", key="sma_name")
    goal = st.radio("Goal", ["increase", "decrease"], horizontal=True, key="sma_goal",
                     help="'increase' for accuracy/relevance; 'decrease' for latency/error rate")

    before, after = data_input_widget("sma")

    if before is not None and after is not None:
        auditor = AIBootstrapAuditor(metric_name, goal=goal)
        result = auditor.audit(before, after, n_iterations=5000)

        if result["verdict"].startswith("GO"):
            st.success(f"**{result['verdict']}**")
        else:
            st.error(f"**{result['verdict']}**")
        st.dataframe(AIBootstrapAuditor.to_dataframe(result), width="stretch")
        st.caption(result["risk_statement"])


# =====================================================================
# TAB 2: GUARDRAIL CHECK (fast path)
# =====================================================================
def guardrail_check_tab():
    st.subheader("Guardrail Check")
    st.caption("Did this regress beyond an acceptable margin? Deliberately more sensitive "
               "than the Audit tab, since a missed regression is riskier than a missed win.")

    metric_name = st.text_input("Guardrail name", value="Latency (ms)", key="grc_name")
    col1, col2 = st.columns(2)
    guard_direction = col1.selectbox(
        "This metric should not:", GUARDRAIL_DIRECTION_OPTIONS,
        format_func=lambda x: GUARDRAIL_DIRECTION_LABELS[x], key="grc_dir",
    )
    margin = col2.number_input("Tolerance margin", min_value=0.0, value=0.0, step=1.0, key="grc_margin",
                                help="0 = must not move at all in the bad direction. "
                                     "Set higher to allow a small, acceptable amount of drift.")

    before, after = data_input_widget("grc")

    if before is not None and after is not None:
        guard = GuardrailAuditor(metric_name, guard_direction=guard_direction, margin=margin)
        result = guard.audit(before, after, n_iterations=5000)

        status_widget = {"PASS": st.success, "FAIL": st.error, "INCONCLUSIVE": st.warning}[result["status"]]
        status_widget(f"**{result['verdict']}**")
        st.dataframe(GuardrailAuditor.to_dataframe(result), width="stretch")
        st.caption(result["risk_statement"])


# =====================================================================
# TAB 3: BATCH REVIEW (fast path)
# =====================================================================
def batch_review_tab():
    st.subheader("Batch Review")
    st.caption("Combine several core metrics and guardrails from the same review cycle into "
               "one consolidated report.")

    with st.expander("Advanced: statistical correction method"):
        st.caption(
            "Testing several metrics at once raises the odds of a false 'GO' by chance alone. "
            "This corrects for that automatically. Benjamini-Hochberg (default) is the right "
            "choice for most reviews; Bonferroni is stricter and better suited to small, "
            "high-stakes batches (roughly 5 metrics or fewer)."
        )
        correction_method = st.selectbox(
            "Method", ["benjamini_hochberg", "bonferroni"], key="batch_method",
            format_func=lambda x: {"benjamini_hochberg": "Benjamini-Hochberg (recommended)",
                                    "bonferroni": "Bonferroni (stricter)"}[x],
        )

    if "batch_metrics" not in st.session_state:
        st.session_state.batch_metrics = []
    if "batch_guardrails" not in st.session_state:
        st.session_state.batch_guardrails = []

    with st.expander("➕ Add a core value metric", expanded=len(st.session_state.batch_metrics) == 0):
        name = st.text_input("Metric name", key="batch_add_metric_name")
        goal = st.radio("Goal", ["increase", "decrease"], horizontal=True, key="batch_add_metric_goal")
        before, after = data_input_widget("batch_metric")
        if st.button("Add metric to batch") and before is not None and name:
            st.session_state.batch_metrics.append(
                {"name": name, "goal": goal, "before": before, "after": after}
            )
            st.success(f"Added '{name}' to batch.")

    with st.expander("➕ Add a guardrail"):
        g_name = st.text_input("Guardrail name", key="batch_add_guard_name")
        g_dir = st.selectbox(
            "This metric should not:", GUARDRAIL_DIRECTION_OPTIONS,
            format_func=lambda x: GUARDRAIL_DIRECTION_LABELS[x], key="batch_add_guard_dir",
        )
        g_margin = st.number_input("Tolerance margin", min_value=0.0, value=0.0, key="batch_add_guard_margin",
                                    help="0 = must not move at all in the bad direction.")
        g_before, g_after = data_input_widget("batch_guard")
        if st.button("Add guardrail to batch") and g_before is not None and g_name:
            st.session_state.batch_guardrails.append(
                {"name": g_name, "guard_direction": g_dir, "margin": g_margin,
                 "before": g_before, "after": g_after}
            )
            st.success(f"Added '{g_name}' to batch.")

    st.write(f"**Batch contents:** {len(st.session_state.batch_metrics)} core metrics, "
             f"{len(st.session_state.batch_guardrails)} guardrails")

    col1, col2 = st.columns([1, 1])
    if col1.button("Run batch", type="primary",
                    disabled=not (st.session_state.batch_metrics or st.session_state.batch_guardrails)):
        batch = AuditBatch(correction_method=correction_method)
        for m in st.session_state.batch_metrics:
            batch.add_metric(m["name"], m["before"], m["after"], goal=m["goal"])
        for g in st.session_state.batch_guardrails:
            batch.add_guardrail(g["name"], g["before"], g["after"],
                                 guard_direction=g["guard_direction"], margin=g["margin"])
        results = batch.run(n_iterations=5000, verbose=False)

        if results:
            st.write("**Core value metrics**")
            st.dataframe(batch.to_dataframe(results), width="stretch")
        guard_df = batch.guardrails_to_dataframe()
        if not guard_df.empty:
            st.write("**Guardrail / cost metrics**")
            st.dataframe(guard_df, width="stretch")
            if (guard_df["Status"] == "INCONCLUSIVE").any():
                st.warning("⚠ One or more guardrails are INCONCLUSIVE -- treat as NOT SAFE "
                           "to ship until reviewed manually or re-tested with more data.")

    if col2.button("Clear batch"):
        st.session_state.batch_metrics = []
        st.session_state.batch_guardrails = []
        st.rerun()


# =====================================================================
# APP LAYOUT
# =====================================================================
st.title("AI Bootstrap Auditor")
st.caption("Is this AI change a real win, a real regression, or just noise? "
           "Paste your before/after results and get a straight answer.")
st.caption("The first three tabs answer that question for a result you already have. "
           "The last two are planning tools for deciding how much data to collect "
           "before you test.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single Metric Audit", "Guardrail Check", "Batch Review",
    "Sample Size Planner", "Harness Calibration (Advanced)",
])

with tab1:
    single_metric_audit_tab()
with tab2:
    guardrail_check_tab()
with tab3:
    batch_review_tab()
with tab4:
    sample_size_planner_section()
with tab5:
    harness_calibration_section()
