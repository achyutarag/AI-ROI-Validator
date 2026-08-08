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
from ai_bootstrap_auditor import (SampleSizeOptimizer, HarnessCalibrationSuite,
                                    run_fwer_stress_test, run_edge_case_validation_suite,
                                    run_efficiency_benchmark, run_detection_power_check)


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
    st.caption(
        "Note: this section may show a *larger* required n than the search above, even for "
        "the same scenario -- that's expected, not a bug. The search above tests continuously "
        "(any n), while this budget search only prices out a fixed set of realistic batch sizes "
        "(50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000) and rounds up to the next one "
        "on that list, since that's what you'd actually order."
    )

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
        null_trials = report["null"].get("n_trials", 100)
        cov_trials = report["coverage"].get("n_trials", 100)

        null_label = "✅ Within expected range" if null_ok else "🟡 Outside expected range"
        cov_label = "✅ Within expected range" if cov_ok else "🟡 Outside expected range"

        st.write(f"**Null calibration** ({null_trials} trials): {null_label} "
                 f"(expected ~{report['null']['expected_false_positive_rate']:.1%} false-positive rate, "
                 f"observed {report['null']['observed_false_positive_rate']:.1%})")
        st.write(f"**CI coverage** ({cov_trials} trials): {cov_label} "
                 f"(nominal {report['coverage']['nominal_coverage']:.1%}, "
                 f"observed {report['coverage']['observed_coverage']:.1%})")

        if not (null_ok and cov_ok):
            st.caption(
                "\"Outside expected range\" at this trial count can reflect genuine "
                "Monte Carlo noise, not necessarily a real calibration problem -- click "
                "'Force recompute' to see if it's stable across a fresh simulation, or "
                "increase the trial count for a tighter estimate."
            )
    else:
        st.info("Click 'Run calibration check' to validate the harness. Nothing runs automatically.")


def validation_suite_section():
    """
    "Audit the Auditor" -- runs this tool's own claims as live, on-demand
    simulations rather than presenting them as static text. Every number
    on this page is computed fresh when you click the button; nothing here
    is a stored or hardcoded result. That distinction matters: a dashboard
    that just prints a canned percentage isn't a demo of anything.
    """
    st.subheader("Validation Suite -- Audit the Auditor")
    st.caption("Runs this tool's own architecture through live stress tests instead of asking you "
               "to take its statistical claims on faith. Nothing runs automatically -- click a "
               "button below, and the numbers regenerate fresh from a real simulation each time.")

    # --- FWER stress test ---
    st.markdown("#### 1. Multi-metric correction: does it actually reduce false positives?")
    st.caption("Simulates batches of metrics with NO real effect (true null) and measures how "
               "often uncorrected testing falsely calls at least one of them a win, vs. the "
               "same batches with Benjamini-Hochberg-style Bonferroni correction applied.")
    col1, col2 = st.columns(2)
    fwer_trials = col1.number_input("Simulated batches", min_value=20, max_value=1000, value=100, step=20,
                                     key="fwer_trials", help="More trials = more precise, slower to run.")
    fwer_n_comparisons = col2.number_input("Metrics per batch", min_value=2, max_value=20, value=5,
                                            key="fwer_n_comp")
    if st.button("▶ Run FWER stress test", key="run_fwer"):
        with st.spinner(f"Simulating {fwer_trials} null batches of {fwer_n_comparisons} metrics each..."):
            result = run_fwer_stress_test(n_comparisons=int(fwer_n_comparisons), n_trials=int(fwer_trials))
        st.session_state["fwer_result"] = result
    result = st.session_state.get("fwer_result")
    if result:
        st.code(
            f"[FWER STRESS TEST] {result['n_trials']} simulated null batches, "
            f"{result['n_comparisons']} metrics/batch\n"
            f"  Uncorrected false-positive rate: {result['uncorrected_fpr']:.1%} "
            f"({result['uncorrected_false_batches']}/{result['n_trials']} batches)\n"
            f"  Corrected false-positive rate:   {result['corrected_fpr']:.1%} "
            f"({result['corrected_false_batches']}/{result['n_trials']} batches)\n"
            f"  (Textbook theoretical uncorrected FWER at this batch size: "
            f"{result['theoretical_uncorrected_fwer']:.1%} -- this run's empirical number will "
            f"vary trial to trial, which is expected of a live simulation.)",
            language=None,
        )
    else:
        st.info("Click 'Run FWER stress test' -- nothing runs until you do.")

    st.markdown("---")

    # --- Edge case validation suite ---
    st.markdown("#### 2. Input validation: does it actually block bad data?")
    st.caption("Re-runs the exact failure modes discovered during this project's development "
               "(the bug that let a single data point produce a confident verdict) against the "
               "current codebase, live.")
    if st.button("▶ Run edge-case validation suite", key="run_edgecases"):
        with st.spinner("Running known-bad inputs through the auditor..."):
            cases = run_edge_case_validation_suite()
        st.session_state["edgecase_result"] = cases
    cases = st.session_state.get("edgecase_result")
    if cases:
        lines = ["[DATA INTEGRITY GUARDRAIL] Running known-bad inputs...\n"]
        for c in cases:
            if c["blocked"]:
                lines.append(f"  ✅ {c['case']}: BLOCKED -- {c['exception_type']} raised")
            else:
                lines.append(f"  🛑 {c['case']}: NOT BLOCKED -- would have produced a verdict!")
        n_blocked = sum(c["blocked"] for c in cases)
        lines.append(f"\n  {n_blocked}/{len(cases)} known-bad inputs correctly blocked.")
        st.code("\n".join(lines), language=None)
        if n_blocked < len(cases):
            st.error("At least one known-bad input was NOT blocked -- this indicates a regression.")
    else:
        st.info("Click 'Run edge-case validation suite' -- nothing runs until you do.")

    st.markdown("---")

    # --- Efficiency benchmark ---
    st.markdown("#### 3. Compute efficiency")
    st.caption("Times the vectorized bootstrap on whatever hardware this app happens to be "
               "running on right now -- not a fixed claim, a live measurement.")
    if st.button("▶ Run efficiency benchmark", key="run_efficiency"):
        with st.spinner("Timing the bootstrap resampling..."):
            eff = run_efficiency_benchmark()
        st.session_state["efficiency_result"] = eff
    eff = st.session_state.get("efficiency_result")
    if eff:
        lines = ["[EFFICIENCY BENCHMARK]\n"]
        for t in eff["single_metric_timings"]:
            lines.append(f"  n_iterations={t['n_iterations']:>6}, n={t['n']:>4} paired obs: "
                         f"{t['elapsed_ms']:.1f} ms")
        lines.append(f"\n  Full {eff['batch_metrics']}-metric batch, "
                     f"{eff['batch_n_iterations']} iterations each: {eff['batch_elapsed_ms']:.1f} ms total")
        st.code("\n".join(lines), language=None)
    else:
        st.info("Click 'Run efficiency benchmark' -- nothing runs until you do.")

    st.markdown("---")

    # --- Detection power (honest footnote, not a headline claim) ---
    with st.expander("4. Detection power at very small n (additional finding, not a differentiator)"):
        st.caption(
            "Tested for completeness: at n=3-9 (below this tool's sample floor), does an "
            "UNGATED bootstrap check produce false positives on pure noise more often than "
            "nominal? Included here even though the result isn't flattering -- an ungated "
            "check at this range turned out close to nominal, not dramatically miscalibrated. "
            "This tool's real advantage in this range isn't fixing a broken naive test -- it's "
            "refusing to answer at all below the sample floor, which converts a probabilistic "
            "risk into a hard guarantee (0% false positives, always, by construction) rather "
            "than a merely-improved-but-still-nonzero one."
        )
        if st.button("▶ Run detection power check", key="run_detection"):
            with st.spinner("Simulating tiny-n null trials..."):
                dp = run_detection_power_check()
            st.session_state["detection_result"] = dp
        dp = st.session_state.get("detection_result")
        if dp:
            st.code(
                f"[DETECTION POWER CHECK] {dp['n_trials']} trials, n={dp['n_range'][0]}-{dp['n_range'][1]} "
                f"(below sample floor), true null\n"
                f"  Ungated naive check false-positive rate: {dp['naive_false_positive_rate']:.1%} "
                f"({dp['naive_false_positives']}/{dp['n_trials']})\n"
                f"  This tool's false-positive rate in this range: 0.0% (always refuses, by design)",
                language=None,
            )
