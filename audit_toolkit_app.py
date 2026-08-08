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
from slow_path_cache import sample_size_planner_section, harness_calibration_section, validation_suite_section

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
    "(custom / paste your own)": {"spec": None, "contexts": {"metric", "guardrail"}},
    "Real accuracy win (70% -> 78%, n=500)": {
        "spec": {"seed": 42, "before_p": 0.70, "after_p": 0.78, "n": 500, "kind": "binomial"},
        "contexts": {"metric"},  # "win" framing doesn't map onto guardrail language well
    },
    "Borderline effect (70% -> 74%, n=80)": {
        "spec": {"seed": 7, "before_p": 0.70, "after_p": 0.74, "n": 80, "kind": "binomial"},
        "contexts": {"metric"},
    },
    "No real effect (70% -> 70%, n=300)": {
        "spec": {"seed": 11, "before_p": 0.70, "after_p": 0.70, "n": 300, "kind": "binomial"},
        "contexts": {"metric"},
    },
    "Latency improvement (1200ms -> 950ms, n=150)": {
        "spec": {"seed": 42, "before_mean": 1200, "after_mean": 950, "scale": 275, "n": 150, "kind": "normal"},
        "contexts": {"metric", "guardrail"},  # a natural fit for both
    },
    "Token count regression (180 -> 214, n=250)": {
        "spec": {"seed": 11, "before_mean": 180, "after_mean": 214, "scale": 42, "n": 250, "kind": "normal"},
        "contexts": {"metric", "guardrail"},
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


COLUMN_HELP = {
    "Sample Size": "Number of paired before/after observations used in this result.",
    "Baseline Mean": "The average value before the change.",
    "After Mean": "The average value after the change.",
    "Lift": "The change from baseline to after (After Mean minus Baseline Mean).",
    "CI Lower": "95% confidence range, low end. The true effect is very likely between "
                "'CI Lower' and 'CI Upper' -- if that range crosses zero, the result can't "
                "be distinguished from random noise.",
    "CI Upper": "95% confidence range, high end. See 'CI Lower' for the full explanation.",
    "Margin": "The allowed amount of drift in the bad direction before this counts as a regression.",
    "p-value": "The probability this result could happen by chance alone if there were truly "
               "no effect. Smaller = stronger evidence of a real effect (below 0.05 is the "
               "common bar). The Verdict column already applies this bar for you.",
    "q-value": "Like p-value, but adjusted for testing several metrics at once "
               "(Benjamini-Hochberg correction) -- this is the number to trust when multiple "
               "metrics were reviewed together in one batch.",
}


def render_results_table(df, single_metric=False):
    """Renders a results dataframe with hover tooltips on the columns that
    carry statistical meaning, so the table is self-explanatory without
    needing someone to walk a reader through it -- e.g. if it gets
    screenshotted into someone else's slide deck.

    single_metric: if True, a None q-value (expected here -- q-values only
    apply when multiple metrics are corrected together) is shown as a plain
    explanation instead of a bare "None", which reads as a bug rather than
    an intentional absence."""
    df = df.copy()
    if single_metric and "q-value" in df.columns:
        df["q-value"] = df["q-value"].apply(
            lambda v: "N/A -- only one metric tested" if pd.isna(v) else v
        )
    column_config = {
        col: st.column_config.Column(help=COLUMN_HELP[col])
        for col in df.columns if col in COLUMN_HELP
    }
    st.dataframe(df, width="stretch", column_config=column_config)


MIN_PAIRED_OBSERVATIONS = 10  # mirrors ai_bootstrap_auditor.MIN_PAIRED_OBSERVATIONS


def _validate_and_clean(before, after) -> tuple:
    """
    Single choke point every data source (example/CSV/paste) and every tab
    (1-3) funnels through before data reaches an auditor. Catches the same
    failure modes as the backend's own guard (too few points, NaN/Inf,
    non-numeric values) but here so the user sees a clear message instead
    of either a raw traceback or -- worse -- a confident-looking verdict
    built on garbage data.
    """
    before = pd.to_numeric(pd.Series(before), errors="coerce").to_numpy(dtype=float)
    after = pd.to_numeric(pd.Series(after), errors="coerce").to_numpy(dtype=float)

    if len(before) != len(after):
        st.error(f"Before ({len(before)} values) and after ({len(after)} values) must be paired -- same length.")
        return None, None

    finite_mask = np.isfinite(before) & np.isfinite(after)
    if not finite_mask.all():
        n_dropped = int((~finite_mask).sum())
        st.warning(f"Dropped {n_dropped} row(s) that were missing, non-numeric, or infinite.")
        before, after = before[finite_mask], after[finite_mask]

    if len(before) < MIN_PAIRED_OBSERVATIONS:
        st.error(
            f"Only {len(before)} usable paired observation(s). This tool exists to catch false "
            f"confidence from small samples, so it won't issue a verdict below "
            f"{MIN_PAIRED_OBSERVATIONS} paired observations -- add more data to continue."
        )
        return None, None

    return before, after


def _seed_control(key_prefix: str) -> int:
    """
    Makes bootstrap results reproducible by default (same inputs -> same
    verdict and CI every time) instead of drawing fresh random samples on
    every click. The re-roll button lets you deliberately re-draw with a
    new seed, which is the right way to sanity-check whether a borderline
    verdict is stable or just landed on one side of a coin flip.
    """
    seed_key = f"{key_prefix}_seed"
    if seed_key not in st.session_state:
        st.session_state[seed_key] = 42
    st.button(
        "🎲 Re-roll (fresh resample)", key=f"{key_prefix}_reroll",
        help="Results are reproducible by default (same data in = same verdict out). "
             "Click to re-draw with a new random seed -- useful to confirm a borderline "
             "verdict isn't just resampling noise landing one way.",
        on_click=lambda: st.session_state.__setitem__(seed_key, st.session_state[seed_key] + 1),
    )
    return st.session_state[seed_key]


def data_input_widget(key_prefix: str, context: str = "metric") -> tuple:
    """
    Returns (before_array, after_array) or (None, None) if not yet provided
    or if the data fails validation. Three input methods: example scenario,
    CSV upload (columns 'before','after'), or manual comma-separated paste.
    Shared across tabs 1-3 so behavior -- and validation -- is consistent
    everywhere data is needed.

    context: "metric" (Single Metric Audit / Batch core metrics) or
    "guardrail" (Guardrail Check / Batch guardrails) -- filters which
    example scenarios are offered, so a scenario framed as a "win" isn't
    shown next to a "should not go up/down" guardrail question.
    """
    method = st.radio(
        "Data source", ["Example scenario", "Upload CSV", "Paste values"],
        horizontal=True, key=f"{key_prefix}_method",
    )

    if method == "Example scenario":
        available = {name: v for name, v in EXAMPLE_SCENARIOS.items() if context in v["contexts"]}
        choice = st.selectbox("Scenario", list(available.keys()), key=f"{key_prefix}_scenario")
        spec = available[choice]["spec"]
        if spec is None:
            st.info("Select a scenario, or switch to Upload/Paste for real data.")
            return None, None
        before, after = _generate_example(spec)

    elif method == "Upload CSV":
        st.caption("CSV must have two columns named exactly 'before' and 'after', same length, paired by row.")
        uploaded = st.file_uploader("Upload CSV", type="csv", key=f"{key_prefix}_upload")
        if uploaded is None:
            return None, None
        try:
            df = pd.read_csv(uploaded)
            before, after = df["before"].to_numpy(), df["after"].to_numpy()
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

    return _validate_and_clean(before, after)


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
        seed = _seed_control("sma")
        auditor = AIBootstrapAuditor(metric_name, goal=goal)
        try:
            result = auditor.audit(before, after, n_iterations=5000, seed=seed)
        except ValueError as e:
            st.error(str(e))
            return

        status_widget = {"GO": st.success, "NO-GO": st.error, "REVIEW": st.warning}[result["status"]]
        status_widget(f"**{result['verdict']}**")
        render_results_table(AIBootstrapAuditor.to_dataframe(result), single_metric=True)
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
                                help="Enter this in the SAME units as your before/after data -- e.g. "
                                     "if your latency values are in ms, a margin of 50 means 50ms of "
                                     "allowed drift, not 50%. 0 = must not move at all in the bad "
                                     "direction.")

    before, after = data_input_widget("grc", context="guardrail")

    if before is not None and after is not None:
        seed = _seed_control("grc")
        guard = GuardrailAuditor(metric_name, guard_direction=guard_direction, margin=margin)
        try:
            result = guard.audit(before, after, n_iterations=5000, seed=seed)
        except ValueError as e:
            st.error(str(e))
            return

        status_widget = {"PASS": st.success, "FAIL": st.error, "INCONCLUSIVE": st.warning}[result["status"]]
        status_widget(f"**{result['verdict']}**")
        render_results_table(GuardrailAuditor.to_dataframe(result))
        st.caption(result["risk_statement"])


# =====================================================================
# TAB 3: BATCH REVIEW (fast path)
# =====================================================================
def _generate_verified_demo_batch():
    """
    Returns the exact same batch used throughout development and testing
    (the Bonferroni/BH flip on the borderline metric, verified across many
    runs). Every array is generated from a LITERAL fixed seed baked directly
    into this function -- not ambient/reused RandomState, not ".random"
    without a seed. That distinction matters specifically because Streamlit
    reruns this entire script on every interaction (including switching the
    correction-method dropdown): any unseeded random call in a data-loading
    path silently regenerates DIFFERENT data on every rerun, which is
    exactly the kind of bug that can make a borderline flip look unreliable
    when it's actually just re-randomizing the underlying data out from
    under you. Calling this function twice, in the same run or a different
    one, always returns byte-identical arrays.

    IMPORTANT: each before/after pair is drawn from ONE RandomState instance,
    sequentially (before first, then after) -- matching exactly how the
    original verified CSVs were generated. Re-seeding separately for
    before vs. after produces DIFFERENT data than what was actually tested.
    """
    def _binomial_pair(seed, before_p, after_p, n):
        r = np.random.RandomState(seed)
        return r.binomial(1, before_p, n).astype(float), r.binomial(1, after_p, n).astype(float)

    def _normal_pair(seed, before_mean, after_mean, scale, n):
        r = np.random.RandomState(seed)
        before = np.round(r.normal(before_mean, scale, n), 1)
        after = np.round(r.normal(after_mean, scale, n), 1)
        return before, after

    m1_b, m1_a = _binomial_pair(121, 0.70, 0.85, 300)
    m2_b, m2_a = _binomial_pair(122, 0.70, 0.705, 300)
    m3_b, m3_a = _binomial_pair(123, 0.50, 0.49, 300)
    m4_b, m4_a = _binomial_pair(124, 0.70, 0.755, 120)   # the borderline flip case
    m5_b, m5_a = _binomial_pair(125, 0.60, 0.605, 300)

    metrics = [
        {"name": "Strong_Win",           "goal": "increase", "before": m1_b, "after": m1_a},
        {"name": "No_Effect_A",          "goal": "increase", "before": m2_b, "after": m2_a},
        {"name": "No_Effect_B",          "goal": "increase", "before": m3_b, "after": m3_a},
        {"name": "Borderline_KEY_CASE",  "goal": "increase", "before": m4_b, "after": m4_a},
        {"name": "No_Effect_C",          "goal": "increase", "before": m5_b, "after": m5_a},
    ]

    g1_b, g1_a = _normal_pair(7, 1200, 900, 80, 100)
    g2_b, g2_a = _normal_pair(7, 1000, 1030, 250, 15)

    guardrails = [
        {"name": "Guardrail_1_Pass",         "guard_direction": "not_increase", "margin": 0.0,
         "before": g1_b, "after": g1_a},
        {"name": "Guardrail_2_Inconclusive", "guard_direction": "not_increase", "margin": 0.0,
         "before": g2_b, "after": g2_a},
    ]
    return metrics, guardrails


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

    with st.expander("🎬 Load verified demo batch (for recording/presenting)"):
        st.caption(
            "Loads the exact 5-metric + 2-guardrail batch used to verify the Bonferroni/BH "
            "flip -- fixed seeds baked into the loader itself, so it reproduces byte-identical "
            "data every click, immune to Streamlit re-running this script on every interaction."
        )
        if st.button("Load verified demo batch", key="load_demo_batch"):
            demo_metrics, demo_guardrails = _generate_verified_demo_batch()
            st.session_state.batch_metrics = demo_metrics
            st.session_state.batch_guardrails = demo_guardrails
            st.success(f"Loaded {len(demo_metrics)} metrics + {len(demo_guardrails)} guardrails.")
            st.rerun()

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
                                    help="Same units as your before/after data (e.g. ms for latency), "
                                         "not a percentage. 0 = must not move at all in the bad direction.")
        g_before, g_after = data_input_widget("batch_guard", context="guardrail")
        if st.button("Add guardrail to batch") and g_before is not None and g_name:
            st.session_state.batch_guardrails.append(
                {"name": g_name, "guard_direction": g_dir, "margin": g_margin,
                 "before": g_before, "after": g_after}
            )
            st.success(f"Added '{g_name}' to batch.")

    st.markdown("##### Batch contents")
    if not st.session_state.batch_metrics and not st.session_state.batch_guardrails:
        st.caption("Nothing added yet -- use the expanders above to build up your batch.")

    if st.session_state.batch_metrics:
        st.caption(f"**Core metrics** ({len(st.session_state.batch_metrics)})")
        for i, m in enumerate(st.session_state.batch_metrics):
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.write(f"**{m['name']}** (goal: {m['goal']})")
            c2.write(f"n = {len(m['before'])}")
            if c3.button("🗑️ Remove", key=f"remove_metric_{i}"):
                st.session_state.batch_metrics.pop(i)
                st.rerun()

    if st.session_state.batch_guardrails:
        st.caption(f"**Guardrails** ({len(st.session_state.batch_guardrails)})")
        for i, g in enumerate(st.session_state.batch_guardrails):
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.write(f"**{g['name']}** (should not {GUARDRAIL_DIRECTION_LABELS[g['guard_direction']]}, "
                     f"margin={g['margin']})")
            c2.write(f"n = {len(g['before'])}")
            if c3.button("🗑️ Remove", key=f"remove_guard_{i}"):
                st.session_state.batch_guardrails.pop(i)
                st.rerun()

    seed = _seed_control("batch")
    col1, col2 = st.columns([1, 1])
    if col1.button("Run batch", type="primary",
                    disabled=not (st.session_state.batch_metrics or st.session_state.batch_guardrails)):
        batch = AuditBatch(correction_method=correction_method)
        for m in st.session_state.batch_metrics:
            batch.add_metric(m["name"], m["before"], m["after"], goal=m["goal"])
        for g in st.session_state.batch_guardrails:
            batch.add_guardrail(g["name"], g["before"], g["after"],
                                 guard_direction=g["guard_direction"], margin=g["margin"])
        try:
            results = batch.run(n_iterations=5000, verbose=False, seed=seed)
        except ValueError as e:
            st.error(f"Couldn't run batch: {e}")
            results = None

        if results:
            st.write("**Core value metrics**")
            render_results_table(batch.to_dataframe(results))
        guard_df = batch.guardrails_to_dataframe()
        if not guard_df.empty:
            st.write("**Guardrail / cost metrics**")
            render_results_table(guard_df)

        # --- Overall recommendation banner ---
        # Synthesizes every core metric + guardrail into one explicit
        # verdict with the specific reason behind it, rather than leaving
        # the reader to scan two tables and infer a conclusion themselves.
        # Priority, worst-first: a FAILED guardrail or a confidently
        # regressed core metric is a hard NO-GO regardless of anything
        # else; missing evidence anywhere is a HOLD, not a silent pass;
        # only a clean sweep is a GO.
        core_statuses = [r["status"] for r in (results or [])]
        guard_statuses = list(guard_df["Status"]) if not guard_df.empty else []

        if "FAIL" in guard_statuses:
            st.error("🔴 **Overall: NO-GO.** A guardrail confidently regressed beyond its allowed "
                      "margin -- do not ship until that's addressed.")
        elif "NO-GO" in core_statuses:
            st.error("🔴 **Overall: NO-GO.** A core metric confidently regressed -- do not ship "
                      "until that's addressed.")
        elif "INCONCLUSIVE" in guard_statuses or "REVIEW" in core_statuses:
            st.warning("🟡 **Overall: HOLD -- decision pending.** No confirmed regressions, but one "
                       "or more metrics don't have enough evidence yet. Collect more data or retest "
                       "before shipping.")
        elif core_statuses or guard_statuses:
            st.success("🟢 **Overall: GO.** No regressions detected across any guardrail, and any "
                       "core metrics tested show a confirmed improvement.")

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
           "The last three are planning and self-validation tools -- deciding how much "
           "data to collect, and checking the tool's own statistical claims hold up.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Single Metric Audit", "Guardrail Check", "Batch Review",
    "Sample Size Planner", "Harness Calibration (Advanced)", "Validation Suite",
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
with tab6:
    validation_suite_section()
