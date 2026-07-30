"""
Caching layer for the SLOW-PATH tools (SampleSizeOptimizer,
HarnessCalibrationSuite). Kept in its own module, separate from
ai_bootstrap_auditor.py, so the core statistical code has zero dependency
on Streamlit and can be tested/reused outside an app.

DESIGN, per the earlier agreed split:
  - Fast path (AIBootstrapAuditor, AuditBatch, GuardrailAuditor, in
    ai_bootstrap_auditor.py + audit_toolkit_app.py): called directly,
    every click, no caching needed -- sub-second already.
  - Slow path (this file -- SampleSizeOptimizer, HarnessCalibrationSuite):
    NEVER called on initial page load. Only runs when a user explicitly
    clicks a "Compute" button, and is cached by st.cache_data keyed on its
    exact inputs, so repeated views of the same scenario don't re-trigger
    a slow simulation.

Sections rendered here:
  - sample_size_planner_section(): required-n search (adaptive), plus a
    "Translate to Budget" block using cached_cost_optimal_plan() -- the
    dollar-cost version of the same search, which is the actual
    exec-facing deliverable ("Go/No-Go investment matrix") from the
    original project pitch.
  - harness_calibration_section(): validates the test's own long-run
    behavior (false-positive rate, CI coverage). Advanced/optional.

CACHE-BUSTING: st.cache_data keys purely on function arguments. To let a
user force a genuine recompute of the SAME inputs (e.g. they suspect the
first run was noisy and want a fresh draw), each cached function takes an
explicit `cache_bust` integer that has no effect on the computation itself
but changes the cache key when incremented -- bumped by the "Recompute"
button via a per-scenario counter in st.session_state.
"""

import streamlit as st
from ai_bootstrap_auditor import SampleSizeOptimizer, HarnessCalibrationSuite


# =====================================================================
# CACHED WRAPPERS -- the actual slow simulations, called only on demand
# =====================================================================

@st.cache_data(show_spinner=False)
def cached_cost_optimal_plan(true_p_before: float, effect_size: float, target_power: float,
                              cost_per_query: float, n_trials: int, cache_bust: int = 0) -> dict:
    """
    Cached wrapper around SampleSizeOptimizer.cost_optimal_plan() -- the
    dollar-cost translation of the required-n search. This is the piece
    that turns "you need n=800" into "this will cost about $400 to know
    for sure," which is the actual executive-facing deliverable the tool
    was pitched on ("Go/No-Go investment matrix"), not just the raw n.
    """
    optimizer = SampleSizeOptimizer(n_bootstrap_iterations=1000)
    return optimizer.cost_optimal_plan(
        true_p_before=true_p_before,
        effect_size=effect_size,
        cost_per_query=cost_per_query,
        target_power=target_power,
        n_trials=n_trials,
    )


@st.cache_data(show_spinner=False)
def cached_find_required_n_adaptive(true_p_before: float, effect_size: float,
                                     target_power: float, n_comparisons: int,
                                     start_n: int, max_n: int, n_trials: int,
                                     cache_bust: int = 0) -> dict:
    """
    Cached wrapper around SampleSizeOptimizer.find_required_n_adaptive().
    `cache_bust` is intentionally unused in the computation -- its only
    job is to appear in the cache key, so incrementing it forces a fresh
    simulation even when every other argument is identical.
    """
    optimizer = SampleSizeOptimizer(n_bootstrap_iterations=1000)
    return optimizer.find_required_n_adaptive(
        true_p_before=true_p_before,
        effect_size=effect_size,
        target_power=target_power,
        n_comparisons=n_comparisons,
        start_n=start_n,
        max_n=max_n,
        n_trials=n_trials,
    )


@st.cache_data(show_spinner=False)
def cached_calibration_report(true_p: float, n_questions: int, n_trials_null: int,
                               n_comparisons: int, true_effect_for_coverage: float,
                               n_trials_coverage: int, cache_bust: int = 0) -> dict:
    """
    Cached wrapper around HarnessCalibrationSuite's null_calibration() and
    ci_coverage() (power_curve omitted from the default bundle -- it's the
    most expensive piece and typically only needed when actively planning
    a sample size, which SampleSizeOptimizer already covers; wire it in
    separately if you want it in this bundle too).
    """
    suite = HarnessCalibrationSuite(n_bootstrap_iterations=800)
    null_result = suite.null_calibration(
        true_p=true_p, n_questions=n_questions, n_trials=n_trials_null,
        n_comparisons=n_comparisons,
    )
    coverage_result = suite.ci_coverage(
        true_p_before=true_p, true_effect=true_effect_for_coverage,
        n_questions=n_questions, n_trials=n_trials_coverage,
    )
    return {"null": null_result, "coverage": coverage_result}


# =====================================================================
# UI HELPERS -- the "Recompute" button + session-state cache-bust pattern
# =====================================================================

def sample_size_planner_section():
    """
    Renders the Sample Size Planner tab. NEVER calls the slow function on
    its own initial render -- only in response to the button click.
    """
    st.subheader("Sample Size Planner")
    st.caption("How much data do you need to trust the result? Set your scenario, then click Compute. "
               "Nothing runs until you click -- this can take a few seconds to a minute.")

    col1, col2, col3 = st.columns(3)
    true_p_before = col1.number_input("Current baseline rate", 0.0, 1.0, 0.70, step=0.01)
    effect_size = col2.number_input("Target effect size (+)", 0.0, 1.0, 0.05, step=0.01)
    target_power = col3.number_input("Target power", 0.0, 1.0, 0.80, step=0.05)

    col4, col5, col6 = st.columns(3)
    start_n = col4.number_input("Start n", min_value=10, value=100, step=10)
    max_n = col5.number_input("Max n", min_value=100, value=20000, step=100)
    n_trials = col6.number_input("Trials per point (higher = slower, more precise)",
                                  min_value=10, value=50, step=10)

    # Per-scenario cache-bust counter, keyed by the scenario itself so
    # switching inputs doesn't require remembering to reset the counter.
    scenario_key = f"n_search_{true_p_before}_{effect_size}_{target_power}_{start_n}_{max_n}_{n_trials}"
    bust_key = f"bust_{scenario_key}"
    if bust_key not in st.session_state:
        st.session_state[bust_key] = 0

    button_col, recompute_col = st.columns([1, 1])
    run_clicked = button_col.button("Compute required sample size", type="primary")
    recompute_clicked = recompute_col.button("🔄 Force recompute (fresh simulation)")

    if recompute_clicked:
        st.session_state[bust_key] += 1
        run_clicked = True  # a forced recompute should also display the result

    if run_clicked:
        with st.spinner(f"Simulating detection rates (n_trials={n_trials} per point, may take a moment)..."):
            result = cached_find_required_n_adaptive(
                true_p_before=true_p_before,
                effect_size=effect_size,
                target_power=target_power,
                n_comparisons=1,
                start_n=int(start_n),
                max_n=int(max_n),
                n_trials=int(n_trials),
                cache_bust=st.session_state[bust_key],
            )
        st.session_state[f"result_{scenario_key}"] = result

    result = st.session_state.get(f"result_{scenario_key}")
    if result is not None:
        if result["required_n"] is not None:
            st.success(f"Required sample size: **{result['required_n']}** "
                       f"for {target_power:.0%} power to detect a +{effect_size:.0%} effect.")
        elif result.get("diminishing_returns"):
            st.warning("Diminishing returns -- more data isn't meaningfully helping. "
                       "This effect may be too small to detect economically.")
        elif result.get("hit_max_n"):
            st.error(f"No sample size up to {max_n} reached {target_power:.0%} power. "
                     f"Try raising max_n or targeting a larger effect.")

        st.dataframe(SampleSizeOptimizer.to_dataframe(result), width="stretch")
        st.caption("Note: results are cached per scenario. Click 'Force recompute' for a "
                   "fresh simulation of the same scenario (useful to sanity-check against noise).")
    else:
        st.info("Set your scenario above and click Compute. Nothing runs until you do.")

    st.markdown("---")
    st.markdown("#### 💰 Translate to Budget")
    st.caption("Uses the same scenario above, plus a cost per query, to answer the question "
               "leadership actually asks: what's the minimum spend to know for sure?")

    cost_per_query = st.number_input(
        "Cost per query ($)", min_value=0.0, value=0.50, step=0.05,
        help="Your fully-loaded cost to run one query through the eval -- LLM API calls, "
             "human labeling, infra, whatever applies.",
    )

    cost_scenario_key = f"{scenario_key}_{cost_per_query}"
    if st.button("Compute cost-optimal plan", type="primary"):
        with st.spinner("Translating the sample-size search into a budget figure..."):
            cost_result = cached_cost_optimal_plan(
                true_p_before=true_p_before,
                effect_size=effect_size,
                target_power=target_power,
                cost_per_query=cost_per_query,
                n_trials=int(n_trials),
                cache_bust=st.session_state[bust_key],
            )
        st.session_state[f"cost_result_{cost_scenario_key}"] = cost_result

    cost_result = st.session_state.get(f"cost_result_{cost_scenario_key}")
    if cost_result is not None:
        if cost_result["required_n"] is not None:
            st.success(
                f"Minimum spend to trust this result: **${cost_result['required_cost']:,.2f}** "
                f"({cost_result['required_n']} queries at ${cost_per_query:.2f} each) for "
                f"{target_power:.0%} power to detect a +{effect_size:.0%} effect."
            )
        else:
            st.warning(
                "No sample size in the default search range reached target power at this "
                "effect size -- this effect may be too small to detect economically, or you "
                "may need the adaptive search above to find a larger required n first."
            )
        cost_df = SampleSizeOptimizer.to_dataframe(cost_result)
        st.dataframe(cost_df, width="stretch")
        st.caption("Each row shows the cost of testing at that sample size, so you can see "
                   "the tradeoff between spend and confidence, not just the cutoff.")


def harness_calibration_section():
    """
    Renders the Harness Calibration tab -- same never-run-on-load,
    button-triggered, cached pattern as the planner above.
    """
    st.subheader("Harness Calibration (Advanced)")
    st.caption("Validates the testing method itself, not any single result -- for confirming the "
               "tool's math is trustworthy before relying on it for a real decision. "
               "Most users won't need this tab day-to-day.")

    col1, col2 = st.columns(2)
    true_p = col1.number_input("Baseline rate for calibration check", 0.0, 1.0, 0.70, step=0.01, key="cal_p")
    n_questions = col2.number_input("Sample size to calibrate at", min_value=10, value=200, step=10, key="cal_n")

    bust_key = f"bust_cal_{true_p}_{n_questions}"
    if bust_key not in st.session_state:
        st.session_state[bust_key] = 0

    run_clicked = st.button("Run calibration check", type="primary")
    if st.button("🔄 Force recompute"):
        st.session_state[bust_key] += 1
        run_clicked = True

    if run_clicked:
        with st.spinner("Running null-effect and coverage simulations..."):
            report = cached_calibration_report(
                true_p=true_p, n_questions=int(n_questions), n_trials_null=100,
                n_comparisons=1, true_effect_for_coverage=0.05, n_trials_coverage=100,
                cache_bust=st.session_state[bust_key],
            )
        st.session_state[f"cal_result_{bust_key}"] = report

    report = st.session_state.get(f"cal_result_{bust_key}")
    if report is not None:
        null_ok = report["null"]["well_calibrated"]
        cov_ok = report["coverage"]["well_calibrated"]
        st.write(f"Null calibration: {'✅ OK' if null_ok else '⚠️ MISCALIBRATED'} "
                 f"(expected ~{report['null']['expected_false_positive_rate']:.1%}, "
                 f"observed {report['null']['observed_false_positive_rate']:.1%})")
        st.write(f"CI coverage: {'✅ OK' if cov_ok else '⚠️ MISCALIBRATED'} "
                 f"(nominal {report['coverage']['nominal_coverage']:.1%}, "
                 f"observed {report['coverage']['observed_coverage']:.1%})")
    else:
        st.info("Click 'Run calibration check' to validate the harness. Nothing runs automatically.")
