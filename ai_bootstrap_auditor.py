import numpy as np
import pandas as pd

class AIBootstrapAuditor:
    def __init__(self, metric_name: str, goal: str = "increase", confidence_level: float = 0.95,
                 n_comparisons: int = 1):
        """
        An enterprise-grade statistical auditing class for AI product features.

        Parameters:
        - metric_name: Name of the metric (e.g., "Accuracy", "Latency")
        - goal: "increase" (higher is better) or "decrease" (lower is better)
        - confidence_level: Usually 0.95 (95% confidence)
        - n_comparisons: How many metrics/tests are being evaluated together in this
          audit round (e.g., if 5 teams each test their own metric in the same
          review cycle, set n_comparisons=5 for each). Applies a Bonferroni
          correction to the significance threshold: alpha_effective = alpha / n_comparisons.
          Default 1 (no correction) preserves prior single-test behavior.

          WHY THIS MATTERS: running this auditor independently across many metrics
          at the nominal alpha=0.05 means you should EXPECT roughly one false "GO"
          per ~20 metrics tested, purely by chance -- even if nothing actually
          improved anywhere. This is the same "fake peak" problem the auditor
          exists to catch, just at the multi-metric level instead of the
          multi-sample level. If you're running this across a batch of metrics in
          one review cycle, set n_comparisons to the batch size.
        """
        if goal not in ["increase", "decrease"]:
            raise ValueError("Goal must be either 'increase' or 'decrease'")
        if n_comparisons < 1:
            raise ValueError("n_comparisons must be >= 1")

        self.metric_name = metric_name
        self.goal = goal
        self.confidence_level = confidence_level
        self.n_comparisons = n_comparisons

    def audit(self, before_scores: np.ndarray, after_scores: np.ndarray, n_iterations: int = 10000) -> dict:
        """
        Runs the paired bootstrap resampling simulation and generates an executive report.
        Vectorized: draws all n_iterations resamples in one call instead of looping
        in pure Python, which is the dominant cost at realistic sample sizes.
        """
        before = np.array(before_scores)
        after = np.array(after_scores)

        if len(before) != len(after):
            raise ValueError("Datasets must be paired! Length of 'before' and 'after' must match.")

        n = len(before)
        observed_before_mean = np.mean(before)
        observed_after_mean = np.mean(after)
        observed_lift = observed_after_mean - observed_before_mean

        # --- Vectorized paired resampling (fix #1) ---
        # Draw all (n_iterations x n) index sets at once, rather than looping
        # n_iterations times in Python and calling np.random.choice/np.mean
        # separately each time. Same statistical procedure, ~10-50x faster in
        # practice since the per-iteration overhead of the Python loop is what
        # dominated runtime, not the underlying math.
        indices = np.random.choice(n, size=(n_iterations, n), replace=True)
        resampled_before = before[indices]   # shape: (n_iterations, n)
        resampled_after = after[indices]     # shape: (n_iterations, n)
        bootstrap_lifts = resampled_after.mean(axis=1) - resampled_before.mean(axis=1)

        # --- Confidence interval ---
        # Bonferroni-adjusted alpha if this audit is one of several run together
        # (fix #2). With n_comparisons=1 this reduces exactly to the original
        # behavior.
        alpha = 1.0 - self.confidence_level
        alpha_effective = alpha / self.n_comparisons
        lower_percentile = (alpha_effective / 2.0) * 100
        upper_percentile = (1.0 - (alpha_effective / 2.0)) * 100

        ci_lower = np.percentile(bootstrap_lifts, lower_percentile)
        ci_upper = np.percentile(bootstrap_lifts, upper_percentile)

        # --- Significance, using the SAME corrected alpha ---
        if self.goal == "increase":
            p_value = np.mean(bootstrap_lifts <= 0)
            is_winner = p_value < alpha_effective and ci_lower > 0
            executive_verdict = "GO (CONFIRMED IMPROVEMENT)" if is_winner else "NO-GO (RISK OF NOISE)"
            risk_statement = f"There is a {p_value * 100:.1f}% chance that this lift is purely background noise."
        else:  # goal == "decrease"
            p_value = np.mean(bootstrap_lifts >= 0)
            is_winner = p_value < alpha_effective and ci_upper < 0
            executive_verdict = "GO (CONFIRMED REDUCTION)" if is_winner else "NO-GO (RISK OF NOISE)"
            risk_statement = f"There is a {p_value * 100:.1f}% chance that this delay/increase is background noise."

        correction_note = (
            f" (Bonferroni-corrected for {self.n_comparisons} simultaneous comparisons: "
            f"effective threshold {alpha_effective:.4f} instead of {alpha:.4f})"
            if self.n_comparisons > 1 else ""
        )

        return {
            "metric": self.metric_name,
            "goal": self.goal,
            "sample_size": n,
            "observed_before": observed_before_mean,
            "observed_after": observed_after_mean,
            "observed_lift": observed_lift,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "alpha_effective": alpha_effective,
            "n_comparisons": self.n_comparisons,
            "verdict": executive_verdict,
            "risk_statement": risk_statement + correction_note,
        }

    @staticmethod
    def to_dataframe(results: dict) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_executive_report().
        Returns a one-row DataFrame so a UI (Streamlit, etc.) can render a
        real table/traffic-light widget instead of parsing terminal text
        via stdout redirection.
        """
        return pd.DataFrame([{
            "Metric": results["metric"],
            "Verdict": results["verdict"],
            "Sample Size": results["sample_size"],
            "Baseline Mean": results["observed_before"],
            "New Mean": results["observed_after"],
            "Lift": results["observed_lift"],
            "CI Lower": results["ci_lower"],
            "CI Upper": results["ci_upper"],
            "p-value": results["p_value"],
            "q-value": results.get("q_value"),  # present only in BH-routed results
        }])

    def print_executive_report(self, results: dict):
        """
        Formats raw statistics into a clear non-technical business evaluation matrix.
        """
        print("=" * 65)
        print(f" EXECUTIVE EVALUATION REPORT: {results['metric'].upper()}")
        print("=" * 65)
        print(f"  VERDICT         : {results['verdict']}")
        print(f"  Tested Sample   : {results['sample_size']} paired queries")
        print(f"  Baseline Mean   : {results['observed_before']:.4f}")
        print(f"  New Mean        : {results['observed_after']:.4f}")
        print(f"  Estimated Change: {results['observed_lift']:+.4f}")
        if results["n_comparisons"] > 1:
            print(f"  95% CI (Bonferroni-adjusted for {results['n_comparisons']} tests): "
                  f"[{results['ci_lower']:+.4f} to {results['ci_upper']:+.4f}]")
        else:
            print(f"  95% Confidence  : [{results['ci_lower']:+.4f} to {results['ci_upper']:+.4f}]")
        print("-" * 65)
        print(f"  ANALYSIS: {results['risk_statement']}")
        print("=" * 65 + "\n")


class AuditBatch:
    """
    Orchestrates multiple AIBootstrapAuditor runs as a single review cycle,
    automatically applying a multiple-comparisons correction to every
    metric in the batch.

    WHY THIS EXISTS: a per-metric correction (like Bonferroni) only works if
    every caller remembers to set n_comparisons correctly by hand -- which is
    exactly the kind of thing that gets silently forgotten in practice (a
    PM adds a 6th metric to this quarter's review and doesn't realize every
    existing AIBootstrapAuditor instantiation needs its n_comparisons bumped
    too). AuditBatch removes that failure mode: you register every metric
    being evaluated in this cycle, and it derives the correction from the
    batch automatically, once, in one place.

    correction_method:
      - "bonferroni" (default): controls the family-wise error rate. Simple,
        conservative -- guarantees the overall false-positive rate across the
        whole batch stays at or below alpha, but gets stricter (harder to
        reach significance) the more metrics you add, even for metrics with
        real effects. Correctly widens each metric's own confidence interval,
        so the reported CI stays simultaneously valid across the batch.
      - "benjamini_hochberg": controls the FALSE DISCOVERY RATE instead --
        the expected proportion of "GO" verdicts that are false positives,
        among all "GO" verdicts. Substantially less conservative than
        Bonferroni as batch size grows, at the cost of a weaker guarantee
        (it bounds the *rate* of false discoveries, not the probability of
        *any* false discovery). Recommended once batches regularly exceed
        ~10 metrics, where Bonferroni starts rejecting real effects it
        shouldn't. NOTE: in this mode, the reported confidence interval for
        each metric is the RAW, uncorrected interval at the nominal
        confidence level -- Benjamini-Hochberg adjusts significance
        decisions (via q-values), not confidence intervals, so the CIs shown
        are not simultaneously valid across the batch the way Bonferroni's
        are. This is flagged explicitly in the report rather than silently
        presenting an interval that looks corrected but isn't.
    """
    VALID_METHODS = ("bonferroni", "benjamini_hochberg")

    def __init__(self, confidence_level: float = 0.95, correction_method: str = "bonferroni"):
        if correction_method not in self.VALID_METHODS:
            raise ValueError(f"correction_method must be one of {self.VALID_METHODS}")
        self.confidence_level = confidence_level
        self.correction_method = correction_method
        self._metrics = []    # core value metrics: list of dicts
        self._guardrails = []  # guardrail metrics: list of dicts (separate family, own routing)

    def add_metric(self, metric_name: str, before_scores, after_scores, goal: str = "increase"):
        """
        Register one metric's paired before/after data into this review
        cycle. Does not run the audit yet -- just queues it, so the total
        batch size (and therefore the correction) is known before any
        individual audit runs.
        """
        if goal not in ["increase", "decrease"]:
            raise ValueError("Goal must be either 'increase' or 'decrease'")
        self._metrics.append({
            "metric_name": metric_name,
            "goal": goal,
            "before": np.array(before_scores),
            "after": np.array(after_scores),
        })
        return self  # allow chaining: batch.add_metric(...).add_metric(...)

    def add_guardrail(self, metric_name: str, before_scores, after_scores,
                       guard_direction: str = "not_increase", margin: float = 0.0,
                       alpha: float = 0.05):
        """
        Register a guardrail/cost metric (latency, tokens, $ per call) --
        a SEPARATE family from core value metrics, routed through
        GuardrailAuditor rather than the correction method used for
        add_metric(). This is deliberate: guardrails need high sensitivity
        to catch regressions, not the conservative correction applied to
        core value metrics. See GuardrailAuditor's docstring for why mixing
        the two families under one correction is wrong.

        alpha here is NOT touched by this batch's correction_method or
        n_comparisons -- guardrails are evaluated independently of how many
        core metrics are in the batch, and independently of each other,
        by default. Pass alpha explicitly per-guardrail if you want to
        tighten a specific one.
        """
        if guard_direction not in ("not_increase", "not_decrease"):
            raise ValueError("guard_direction must be 'not_increase' or 'not_decrease'")
        self._guardrails.append({
            "metric_name": metric_name,
            "guard_direction": guard_direction,
            "margin": margin,
            "alpha": alpha,
            "before": np.array(before_scores),
            "after": np.array(after_scores),
        })
        return self  # allow chaining

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float):
        """
        Standard BH step-up procedure. Returns (significant_mask, q_values),
        both in the ORIGINAL input order (not sorted).
        """
        m = len(p_values)
        order = np.argsort(p_values)
        sorted_p = p_values[order]

        # Largest k such that sorted_p[k-1] <= (k/m) * alpha  (1-indexed k)
        thresholds = (np.arange(1, m + 1) / m) * alpha
        passing = sorted_p <= thresholds
        if passing.any():
            k = np.max(np.where(passing)[0]) + 1  # 1-indexed count of rejections
        else:
            k = 0

        significant_sorted = np.zeros(m, dtype=bool)
        significant_sorted[:k] = True

        # q-values: step-up monotone adjustment, q_(i) = min_{j>=i} (p_(j) * m / j)
        raw_q = sorted_p * m / np.arange(1, m + 1)
        q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
        q_sorted = np.clip(q_sorted, 0, 1)

        # Map back to original order
        significant = np.empty(m, dtype=bool)
        q_values = np.empty(m, dtype=float)
        significant[order] = significant_sorted
        q_values[order] = q_sorted
        return significant, q_values

    def run(self, n_iterations: int = 10000, verbose: bool = True) -> list:
        """
        Runs every registered core metric's audit, applying whichever
        correction_method was set at construction, AND every registered
        guardrail's audit via GuardrailAuditor (own routing, not affected
        by correction_method). Returns the list of CORE metric results
        (unchanged shape/behavior from before guardrails existed); guardrail
        results are stored internally and included automatically by
        print_batch_summary().
        """
        if not self._metrics and not self._guardrails:
            raise ValueError("No metrics or guardrails registered. Call add_metric() "
                              "and/or add_guardrail() before run().")

        results = []
        if self._metrics:
            n_comparisons = len(self._metrics)
            alpha = 1.0 - self.confidence_level

            if self.correction_method == "bonferroni":
                for spec in self._metrics:
                    auditor = AIBootstrapAuditor(
                        metric_name=spec["metric_name"],
                        goal=spec["goal"],
                        confidence_level=self.confidence_level,
                        n_comparisons=n_comparisons,
                    )
                    result = auditor.audit(spec["before"], spec["after"], n_iterations=n_iterations)
                    results.append(result)
                    if verbose:
                        auditor.print_executive_report(result)

            else:  # benjamini_hochberg
                raw_results = []
                for spec in self._metrics:
                    auditor = AIBootstrapAuditor(
                        metric_name=spec["metric_name"],
                        goal=spec["goal"],
                        confidence_level=self.confidence_level,
                        n_comparisons=1,
                    )
                    raw_results.append(auditor.audit(spec["before"], spec["after"], n_iterations=n_iterations))

                p_values = np.array([r["p_value"] for r in raw_results])
                significant, q_values = self._benjamini_hochberg(p_values, alpha)

                for r, spec, is_sig, q in zip(raw_results, self._metrics, significant, q_values):
                    r = dict(r)
                    r["q_value"] = q
                    r["bh_significant"] = bool(is_sig)
                    r["n_comparisons"] = n_comparisons
                    direction_word = "IMPROVEMENT" if spec["goal"] == "increase" else "REDUCTION"
                    r["verdict"] = f"GO (CONFIRMED {direction_word})" if is_sig else "NO-GO (RISK OF NOISE)"
                    r["risk_statement"] = (
                        f"Benjamini-Hochberg adjusted p-value (q-value): {q:.4f} "
                        f"(raw p-value: {r['p_value']:.4f}, batch size: {n_comparisons}). "
                        f"NOTE: confidence interval below is the RAW nominal interval, "
                        f"not simultaneously corrected -- BH adjusts the significance "
                        f"decision, not the interval."
                    )
                    results.append(r)
                    if verbose:
                        auditor_for_print = AIBootstrapAuditor(spec["metric_name"], spec["goal"])
                        auditor_for_print.print_executive_report(r)

        # --- Guardrails: separate routing, always independent of correction_method ---
        guardrail_results = []
        for spec in self._guardrails:
            guard = GuardrailAuditor(
                metric_name=spec["metric_name"],
                guard_direction=spec["guard_direction"],
                margin=spec["margin"],
                alpha=spec["alpha"],
            )
            g_result = guard.audit(spec["before"], spec["after"], n_iterations=n_iterations)
            guardrail_results.append(g_result)
            if verbose:
                guard.print_executive_report(g_result)

        self._last_guardrail_results = guardrail_results
        return results

    def to_dataframe(self, results: list) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_batch_summary()'s
        core-metrics table. One row per metric.
        """
        rows = []
        for r in results:
            row = {
                "Metric": r["metric"],
                "Verdict": r["verdict"],
                "Lift": r["observed_lift"],
                "CI Lower": r["ci_lower"],
                "CI Upper": r["ci_upper"],
            }
            if self.correction_method == "benjamini_hochberg":
                row["q-value"] = r.get("q_value")
            else:
                row["p-value"] = r.get("p_value")
            rows.append(row)
        return pd.DataFrame(rows)

    def guardrails_to_dataframe(self) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_batch_summary()'s
        guardrail section. Must be called after run(); reads the results
        stored by the most recent run() call.
        """
        guardrail_results = getattr(self, "_last_guardrail_results", [])
        rows = [{
            "Metric": r["metric"],
            "Status": r["status"],
            "Lift": r["observed_lift"],
            "Margin": r["margin"],
            "Lower Bound": r["lower_bound"],
            "Upper Bound": r["upper_bound"],
        } for r in guardrail_results]
        return pd.DataFrame(rows)

    def print_batch_summary(self, results: list):
        """
        A single consolidated table across every metric in the batch --
        the actual "executive report" a PM would want to scan in one
        glance, rather than reading N separate full reports. Automatically
        includes a second section for guardrail results if any were
        registered via add_guardrail() and run() has been called.
        """
        if results:
            n_comparisons = len(results)
            print("#" * 65)
            print(f" CORE VALUE METRICS: {n_comparisons} metrics tested this cycle")
            print(f" Correction method: {self.correction_method}")
            if self.correction_method == "bonferroni":
                print(f" Bonferroni-corrected threshold applied to each: "
                      f"{(1 - self.confidence_level) / n_comparisons:.4f}")
            else:
                print(f" Benjamini-Hochberg: significance based on q-value < "
                      f"{1 - self.confidence_level:.4f} (false discovery rate control)")
            print("#" * 65)
            if self.correction_method == "benjamini_hochberg":
                print(f"  {'Metric':<28} {'Lift':>12} {'q-value':>10} {'Verdict':>26}")
                print("  " + "-" * 78)
                for r in results:
                    lift_str = f"{r['observed_lift']:+.4f}"
                    q_str = f"{r['q_value']:.4f}"
                    print(f"  {r['metric']:<28} {lift_str:>12} {q_str:>10} {r['verdict']:>26}")
            else:
                print(f"  {'Metric':<28} {'Lift':>12} {'Verdict':>28}")
                print("  " + "-" * 68)
                for r in results:
                    lift_str = f"{r['observed_lift']:+.4f}"
                    print(f"  {r['metric']:<28} {lift_str:>12} {r['verdict']:>28}")
            print("#" * 65 + "\n")

        guardrail_results = getattr(self, "_last_guardrail_results", [])
        if guardrail_results:
            print("#" * 65)
            print(f" GUARDRAIL / COST METRICS: {len(guardrail_results)} checked this cycle")
            print(" Routing: GuardrailAuditor (high-sensitivity, non-inferiority test --")
            print(" NOT the same correction as core metrics above; see GuardrailAuditor")
            print(" docstring for why applying a conservative correction here would be wrong)")
            print("#" * 65)
            print(f"  {'Metric':<38} {'Lift':>10} {'Status':>18}")
            print("  " + "-" * 70)
            for r in guardrail_results:
                lift_str = f"{r['observed_lift']:+.2f}"
                print(f"  {r['metric']:<38} {lift_str:>10} {r['status']:>18}")
            if any(r["status"] == "INCONCLUSIVE" for r in guardrail_results):
                print("\n  ⚠ One or more guardrails are INCONCLUSIVE -- treat as NOT SAFE")
                print("    to ship until reviewed manually or re-tested with more data.")
            print("#" * 65 + "\n")


class GuardrailAuditor:
    """
    Purpose-built for guardrail/cost metrics (latency, tokens, $ per call),
    where the question is NOT "did this improve" but "did this regress
    beyond an acceptable margin" -- and where missing a real regression is
    the expensive mistake, not raising a false alarm.

    THIS IS DELIBERATELY NOT AIBootstrapAuditor WITH A STRICTER THRESHOLD.
    Applying a conservative correction (like Bonferroni) to a guardrail
    metric makes it HARDER to detect a regression, not easier -- a stricter
    significance bar means it takes a bigger, more obvious regression to
    trigger a flag. That is the opposite of what a guardrail should do.
    Guardrails should default to high sensitivity, not high conservatism.

    METHOD: non-inferiority testing via bootstrap, with an explicit
    THIRD outcome (INCONCLUSIVE) instead of collapsing "not enough
    evidence" into "assume it's fine." Treating an inconclusive result as
    a pass is the absence-of-evidence trap -- exactly what would let a
    real, moderate cost regression slip through silently. This auditor
    refuses to do that: it only reports PASS when there is actual
    confidence the regression, if any, is within the allowed margin.

    Parameters:
    - guard_direction: "not_increase" (e.g. latency, cost, token count --
      going up is bad) or "not_decrease" (e.g. a safety/accuracy floor --
      going down is bad).
    - margin: how much change is tolerated before it counts as a real
      regression. margin=0 means "must not get worse AT ALL" (strict).
      For token count you might set margin=0 (any real increase is a
      regression); for latency you might set margin=20 (allow up to 20ms
      of drift before flagging).
    - alpha: one-sided significance level for the bounds below. Default
      0.05 (95% one-sided confidence). Deliberately NOT tightened by a
      multiple-comparisons correction by default -- see class docstring.
      If you are genuinely running many guardrail checks at once and want
      to control false alarms across them, you may still pass a smaller
      alpha manually, but the default here favors catching regressions
      over avoiding false alarms, which is the right default for a
      guardrail.
    """
    def __init__(self, metric_name: str, guard_direction: str = "not_increase",
                 margin: float = 0.0, alpha: float = 0.05):
        if guard_direction not in ("not_increase", "not_decrease"):
            raise ValueError("guard_direction must be 'not_increase' or 'not_decrease'")
        if margin < 0:
            raise ValueError("margin must be >= 0 (it's a tolerance magnitude, sign is handled internally)")
        self.metric_name = metric_name
        self.guard_direction = guard_direction
        self.margin = margin
        self.alpha = alpha

    def audit(self, before_scores, after_scores, n_iterations: int = 10000) -> dict:
        before = np.array(before_scores)
        after = np.array(after_scores)
        if len(before) != len(after):
            raise ValueError("Datasets must be paired! Length of 'before' and 'after' must match.")

        n = len(before)
        observed_lift = np.mean(after) - np.mean(before)

        # Vectorized bootstrap (same approach as AIBootstrapAuditor's fix #1)
        indices = np.random.choice(n, size=(n_iterations, n), replace=True)
        bootstrap_lifts = after[indices].mean(axis=1) - before[indices].mean(axis=1)

        # One-sided bounds at the SAME alpha in both directions -- we need
        # both bounds regardless of guard_direction, because we use one to
        # test for PASS and the other to test for FAIL; whichever isn't the
        # "active" side just tells us how much headroom/margin exists.
        lower_bound = np.percentile(bootstrap_lifts, self.alpha * 100)
        upper_bound = np.percentile(bootstrap_lifts, (1 - self.alpha) * 100)

        if self.guard_direction == "not_increase":
            # Bad = lift > margin. PASS if we're confident lift <= margin
            # (upper bound doesn't exceed margin). FAIL if we're confident
            # lift > margin (lower bound already exceeds margin).
            if upper_bound <= self.margin:
                status = "PASS"
            elif lower_bound > self.margin:
                status = "FAIL"
            else:
                status = "INCONCLUSIVE"
            threshold_desc = f"lift must not confidently exceed +{self.margin}"
        else:  # not_decrease
            # Bad = lift < -margin. PASS if confident lift >= -margin
            # (lower bound doesn't go below -margin). FAIL if confident
            # lift < -margin (upper bound is still below -margin).
            if lower_bound >= -self.margin:
                status = "PASS"
            elif upper_bound < -self.margin:
                status = "FAIL"
            else:
                status = "INCONCLUSIVE"
            threshold_desc = f"lift must not confidently fall below -{self.margin}"

        verdict_map = {
            "PASS": "PASS (NO MEANINGFUL REGRESSION)",
            "FAIL": "FAIL (REGRESSION DETECTED)",
            "INCONCLUSIVE": "INCONCLUSIVE (NOT ENOUGH EVIDENCE -- DO NOT ASSUME SAFE)",
        }

        risk_statement = {
            "PASS": f"We are >= {(1 - self.alpha) * 100:.0f}% confident this metric's change stays within "
                    f"the allowed margin ({threshold_desc}).",
            "FAIL": f"We are >= {(1 - self.alpha) * 100:.0f}% confident this metric regressed beyond "
                    f"the allowed margin ({threshold_desc}).",
            "INCONCLUSIVE": f"The data cannot confirm the change stays within margin, but also cannot "
                             f"confirm a regression. This is NOT a pass -- gather more data or review "
                             f"manually before shipping.",
        }[status]

        return {
            "metric": self.metric_name,
            "guard_direction": self.guard_direction,
            "margin": self.margin,
            "alpha": self.alpha,
            "sample_size": n,
            "observed_before": np.mean(before),
            "observed_after": np.mean(after),
            "observed_lift": observed_lift,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "status": status,
            "verdict": verdict_map[status],
            "risk_statement": risk_statement,
        }

    @staticmethod
    def to_dataframe(results: dict) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_executive_report().
        """
        return pd.DataFrame([{
            "Metric": results["metric"],
            "Status": results["status"],
            "Sample Size": results["sample_size"],
            "Baseline Mean": results["observed_before"],
            "New Mean": results["observed_after"],
            "Lift": results["observed_lift"],
            "Margin": results["margin"],
            "Lower Bound": results["lower_bound"],
            "Upper Bound": results["upper_bound"],
        }])

    def print_executive_report(self, results: dict):
        print("=" * 65)
        print(f" GUARDRAIL REPORT: {results['metric'].upper()}")
        print("=" * 65)
        print(f"  VERDICT         : {results['verdict']}")
        print(f"  Tested Sample   : {results['sample_size']} paired queries")
        print(f"  Baseline Mean   : {results['observed_before']:.4f}")
        print(f"  New Mean        : {results['observed_after']:.4f}")
        print(f"  Estimated Change: {results['observed_lift']:+.4f}  (margin: {results['margin']})")
        print(f"  One-sided bounds: [{results['lower_bound']:+.4f} to {results['upper_bound']:+.4f}] "
              f"at {(1 - results['alpha']) * 100:.0f}% confidence")
        print("-" * 65)
        print(f"  ANALYSIS: {results['risk_statement']}")
        print("=" * 65 + "\n")


class HarnessCalibrationSuite:
    """
    Validates the STATISTICAL MACHINERY itself, not any one result -- the
    correct analog to the compliance guard's fit/held-out/adversarial
    protocol, adapted to what's actually at risk here.

    IMPORTANT DISTINCTION FROM THE COMPLIANCE GUARD:
    The compliance guard had a trained decision rule (thresholds, a
    classifier) that could overfit to the exact data used to build it --
    held-out splits existed to catch that circularity. This harness has no
    trained rule to overfit; bootstrap resampling and the BH/Bonferroni math
    are generic procedures applied fresh to whatever data comes in. There is
    nothing to "hold out." Individual NO-GO or INCONCLUSIVE results are not
    noise to be filtered out -- they are the honest output of a correctly
    functioning test given limited evidence, and should not be "fixed."

    What CAN and SHOULD be validated is whether the test's own long-run
    behavior matches what it claims:
      1. Null calibration: when there is truly NO effect, does the false
         positive rate actually match alpha (e.g. ~5%)? If not, the test
         itself is miscalibrated -- a real defect worth catching.
      2. Power curve: when there IS a real effect of a given size, how often
         does the test actually detect it, at various sample sizes? This is
         planning information ("at n=400 with 2 simultaneous comparisons, an
         8-point true effect is only caught in ~40% of trials"), not a
         correctness check on any single result.
      3. CI coverage: across many simulated trials with a known true lift,
         does the reported 95% CI actually contain the true value ~95% of
         the time?
    """
    def __init__(self, confidence_level: float = 0.95, n_bootstrap_iterations: int = 2000):
        self.confidence_level = confidence_level
        self.n_bootstrap_iterations = n_bootstrap_iterations

    def null_calibration(self, true_p: float = 0.70, n_questions: int = 200,
                          n_trials: int = 200, n_comparisons: int = 1,
                          goal: str = "increase", seed: int = 0) -> dict:
        """
        Simulates n_trials independent before/after pairs where the TRUE
        rate is identical in both (no real effect exists), and checks how
        often the auditor incorrectly says GO. Expected false-positive rate
        is roughly alpha_effective = (1 - confidence_level) / n_comparisons,
        one-sided.
        """
        rng = np.random.RandomState(seed)
        alpha_effective = (1 - self.confidence_level) / n_comparisons
        false_positives = 0

        for _ in range(n_trials):
            before = rng.binomial(1, p=true_p, size=n_questions)
            after = rng.binomial(1, p=true_p, size=n_questions)  # same true rate -- no real effect
            auditor = AIBootstrapAuditor("calibration_check", goal=goal,
                                          confidence_level=self.confidence_level,
                                          n_comparisons=n_comparisons)
            result = auditor.audit(before, after, n_iterations=self.n_bootstrap_iterations)
            if result["verdict"].startswith("GO"):
                false_positives += 1

        observed_rate = false_positives / n_trials
        return {
            "n_trials": n_trials,
            "expected_false_positive_rate": alpha_effective,
            "observed_false_positive_rate": observed_rate,
            "n_false_positives": false_positives,
            "well_calibrated": abs(observed_rate - alpha_effective) < 0.05,  # loose tolerance, Monte Carlo noise
        }

    def power_curve(self, true_p_before: float = 0.70, effect_sizes=(0.02, 0.05, 0.08, 0.10),
                     sample_sizes=(100, 200, 500, 1000), n_trials: int = 100,
                     n_comparisons: int = 1, seed: int = 1) -> list:
        """
        For each (effect_size, sample_size) combination, simulates n_trials
        audits with that TRUE effect actually present, and reports the
        fraction correctly detected as GO. This is the "how much power do I
        actually have" report a PM should read BEFORE trusting any single
        NO-GO as "there's no effect" rather than "we didn't have enough
        data to see it."
        """
        rng = np.random.RandomState(seed)
        rows = []
        for n in sample_sizes:
            for effect in effect_sizes:
                true_p_after = min(true_p_before + effect, 1.0)
                detections = 0
                for _ in range(n_trials):
                    before = rng.binomial(1, p=true_p_before, size=n)
                    after = rng.binomial(1, p=true_p_after, size=n)
                    auditor = AIBootstrapAuditor("power_check", goal="increase",
                                                  confidence_level=self.confidence_level,
                                                  n_comparisons=n_comparisons)
                    result = auditor.audit(before, after, n_iterations=self.n_bootstrap_iterations)
                    if result["verdict"].startswith("GO"):
                        detections += 1
                rows.append({
                    "sample_size": n,
                    "true_effect_size": effect,
                    "n_trials": n_trials,
                    "detection_rate": detections / n_trials,
                })
        return rows

    def ci_coverage(self, true_p_before: float = 0.70, true_effect: float = 0.05,
                     n_questions: int = 200, n_trials: int = 200, seed: int = 2) -> dict:
        """
        Simulates n_trials audits with a KNOWN true lift, and checks what
        fraction of the reported 95% CIs actually contain that true value.
        Should be close to the nominal confidence level (e.g. ~95%). A
        coverage rate far below nominal means the CI is too narrow (overly
        confident); far above means it's too wide (overly conservative).
        """
        rng = np.random.RandomState(seed)
        true_p_after = min(true_p_before + true_effect, 1.0)
        covered = 0

        for _ in range(n_trials):
            before = rng.binomial(1, p=true_p_before, size=n_questions)
            after = rng.binomial(1, p=true_p_after, size=n_questions)
            auditor = AIBootstrapAuditor("coverage_check", goal="increase",
                                          confidence_level=self.confidence_level,
                                          n_comparisons=1)
            result = auditor.audit(before, after, n_iterations=self.n_bootstrap_iterations)
            if result["ci_lower"] <= true_effect <= result["ci_upper"]:
                covered += 1

        observed_coverage = covered / n_trials
        return {
            "n_trials": n_trials,
            "true_effect": true_effect,
            "nominal_coverage": self.confidence_level,
            "observed_coverage": observed_coverage,
            "well_calibrated": abs(observed_coverage - self.confidence_level) < 0.05,
        }

    def print_full_report(self, null_result: dict, power_rows: list, coverage_result: dict):
        print("=" * 70)
        print(" HARNESS CALIBRATION REPORT")
        print(" (validates the TEST ITSELF, not any single audit result)")
        print("=" * 70)

        print("\n[1] Null calibration -- false positive rate when NO real effect exists")
        status = "OK" if null_result["well_calibrated"] else "MISCALIBRATED"
        print(f"    Expected: ~{null_result['expected_false_positive_rate']:.1%}   "
              f"Observed: {null_result['observed_false_positive_rate']:.1%}   "
              f"({null_result['n_false_positives']}/{null_result['n_trials']} trials)   [{status}]")

        print("\n[2] Power curve -- detection rate when a REAL effect of this size exists")
        print(f"    {'n':>6} {'true effect':>12} {'detection rate':>16}")
        for row in power_rows:
            print(f"    {row['sample_size']:>6} {row['true_effect_size']:>12.0%} "
                  f"{row['detection_rate']:>16.0%}")
        print("    -> Read this BEFORE trusting a NO-GO as 'no effect exists.' A low")
        print("       detection rate at your actual sample size means a NO-GO could")
        print("       easily be a real effect the test didn't have power to catch.")

        print("\n[3] CI coverage -- does the reported 95% CI actually contain the truth ~95% of the time?")
        status = "OK" if coverage_result["well_calibrated"] else "MISCALIBRATED"
        print(f"    Nominal: {coverage_result['nominal_coverage']:.1%}   "
              f"Observed: {coverage_result['observed_coverage']:.1%}   [{status}]")
        print("=" * 70 + "\n")


class SampleSizeOptimizer:
    """
    Inverts the power curve: instead of "given n, what's my power?", answers
    "given a target power, what's the minimum n?" -- computed via the SAME
    simulation approach as HarnessCalibrationSuite, not a closed-form formula
    and not a hardcoded lookup table. Every number this class reports is
    something it actually simulated, not an assumption or extrapolation.

    WHY NOT JUST HARDCODE A TABLE: a plausible-sounding table like "n=150-200
    for a 15% effect, n=1500+ for a 2-3% effect" is exactly the kind of
    unverified claim this whole project has been built to avoid trusting
    without evidence. Effect size, true baseline rate, and n_comparisons all
    interact -- a table computed for one true_p_before or one n_comparisons
    value will not automatically hold for another. This class computes the
    actual answer for YOUR specific scenario, every time.
    """
    def __init__(self, confidence_level: float = 0.95, n_bootstrap_iterations: int = 1000):
        self.suite = HarnessCalibrationSuite(confidence_level=confidence_level,
                                              n_bootstrap_iterations=n_bootstrap_iterations)

    def _detection_rate_at_n(self, n: int, true_p_before: float, effect_size: float,
                              n_comparisons: int, n_trials: int, rng: np.random.RandomState) -> float:
        """
        Shared simulation core: runs n_trials audits at sample size n with
        the given true effect actually present, returns fraction detected
        as GO. Used by both find_required_n() (fixed grid) and
        find_required_n_adaptive() (exponential search + refinement), so
        the two methods can't silently drift out of sync with each other.
        """
        true_p_after = min(true_p_before + effect_size, 1.0)
        detections = 0
        for _ in range(n_trials):
            before = rng.binomial(1, p=true_p_before, size=n)
            after = rng.binomial(1, p=true_p_after, size=n)
            auditor = AIBootstrapAuditor("n_search", goal="increase",
                                          confidence_level=self.suite.confidence_level,
                                          n_comparisons=n_comparisons)
            result = auditor.audit(before, after, n_iterations=self.suite.n_bootstrap_iterations)
            if result["verdict"].startswith("GO"):
                detections += 1
        return detections / n_trials

    def find_required_n(self, true_p_before: float, effect_size: float,
                         target_power: float = 0.80, n_comparisons: int = 1,
                         candidate_sizes=(50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000),
                         n_trials: int = 50, seed: int = 3) -> dict:
        """
        Simulates detection rate at each candidate sample size for a
        SPECIFIC true_p_before/effect_size/n_comparisons combination, and
        returns the smallest sample size that empirically reaches
        target_power. Returns the full curve too, so you can see the
        tradeoff, not just the cutoff.

        LIMITATION: only searches the exact sizes you give it. If the true
        required n is above your largest candidate, this returns
        required_n=None with no indication of how much further to look --
        see find_required_n_adaptive() for a version that keeps searching
        automatically.

        NOTE: n_trials controls how precise this estimate is. n_trials=50
        is fast enough to explore interactively but has real Monte Carlo
        noise (+/- ~10-15 percentage points on the detection rate estimate
        at that count) -- rerun with a higher n_trials (200+) before using
        the result to justify a real budget decision.
        """
        rng_seed = seed
        curve = []
        required_n = None

        for n in sorted(candidate_sizes):
            rng = np.random.RandomState(rng_seed)
            rng_seed += 1
            detection_rate = self._detection_rate_at_n(n, true_p_before, effect_size,
                                                         n_comparisons, n_trials, rng)
            curve.append({"sample_size": n, "detection_rate": detection_rate})
            if required_n is None and detection_rate >= target_power:
                required_n = n

        return {
            "true_p_before": true_p_before,
            "effect_size": effect_size,
            "n_comparisons": n_comparisons,
            "target_power": target_power,
            "required_n": required_n,  # None if no candidate size reached target_power
            "curve": curve,
            "n_trials_per_point": n_trials,
        }

    def find_required_n_adaptive(self, true_p_before: float, effect_size: float,
                                  target_power: float = 0.80, n_comparisons: int = 1,
                                  start_n: int = 50, max_n: int = 50000,
                                  growth_factor: float = 2.0, n_trials: int = 50,
                                  refine: bool = True, refine_steps: int = 6,
                                  diminishing_returns_threshold: float = 0.03,
                                  seed: int = 5) -> dict:
        """
        Doesn't require you to guess a candidate range up front. Starts at
        start_n and keeps doubling (by growth_factor) until either:
          (a) detection rate reaches target_power -- search stops, or
          (b) n exceeds max_n -- search stops, reports hit_max_n=True,
              meaning this effect may not be economically detectable, or a
              higher max_n is needed, or
          (c) diminishing returns -- doubling the sample size is barely
              moving the detection rate (delta < diminishing_returns_threshold
              between consecutive doublings) while still well below target
              (< 70% of target_power). This usually means the effect is too
              small relative to the noise floor to be worth chasing with
              more data at all -- search stops early, reports
              diminishing_returns=True, instead of walking all the way to
              max_n for a result that was already predictable. Set
              diminishing_returns_threshold=0 to disable this check.

        If refine=True and a passing n was found, runs a binary search
        between the last failing size and the first passing size to
        tighten the estimate, rather than reporting the coarse doubled
        value (e.g. if 1500 fails and 3000 passes, refine narrows toward
        the actual crossover instead of stopping at 3000).

        Returns the same shape as find_required_n(), plus:
          - "search_path": the exponential-phase (n, detection_rate) pairs
          - "hit_max_n": True if max_n was reached without finding target_power
          - "diminishing_returns": True if the search stopped early because
            more data stopped meaningfully helping
        """
        rng_seed = seed
        search_path = []
        n = start_n
        last_failing_n = None
        first_passing_n = None
        first_passing_rate = None
        hit_max_n = False
        diminishing_returns = False
        previous_detection_rate = None

        # --- Phase 1: exponential search ---
        while True:
            rng = np.random.RandomState(rng_seed)
            rng_seed += 1
            detection_rate = self._detection_rate_at_n(n, true_p_before, effect_size,
                                                         n_comparisons, n_trials, rng)
            search_path.append({"sample_size": n, "detection_rate": detection_rate})

            if detection_rate >= target_power:
                first_passing_n = n
                first_passing_rate = detection_rate
                break

            # Diminishing-returns check: only meaningful once we have at
            # least two points to compare, and only worth stopping early if
            # we're still clearly far from the target (otherwise a small
            # delta near the target is expected and fine -- that's normal
            # convergence, not a sign the effect is undetectable).
            if (previous_detection_rate is not None
                    and diminishing_returns_threshold > 0
                    and detection_rate < target_power * 0.70):
                delta = detection_rate - previous_detection_rate
                if delta < diminishing_returns_threshold:
                    diminishing_returns = True
                    last_failing_n = n
                    break

            previous_detection_rate = detection_rate
            last_failing_n = n
            next_n = int(n * growth_factor)
            if next_n > max_n:
                hit_max_n = True
                break
            n = next_n

        curve = list(search_path)
        required_n = first_passing_n

        # --- Phase 2: optional binary-search refinement between bounds ---
        # (skipped automatically if diminishing_returns/hit_max_n stopped us
        # before ever finding a passing n -- nothing to refine toward)
        if refine and first_passing_n is not None and last_failing_n is not None:
            lo, hi = last_failing_n, first_passing_n
            for _ in range(refine_steps):
                mid = (lo + hi) // 2
                if mid <= lo or mid >= hi:
                    break  # bounds have converged as tightly as integers allow
                rng = np.random.RandomState(rng_seed)
                rng_seed += 1
                mid_rate = self._detection_rate_at_n(mid, true_p_before, effect_size,
                                                       n_comparisons, n_trials, rng)
                curve.append({"sample_size": mid, "detection_rate": mid_rate})
                if mid_rate >= target_power:
                    hi = mid
                    required_n = mid
                else:
                    lo = mid

        curve.sort(key=lambda row: row["sample_size"])

        return {
            "true_p_before": true_p_before,
            "effect_size": effect_size,
            "n_comparisons": n_comparisons,
            "target_power": target_power,
            "required_n": required_n,
            "hit_max_n": hit_max_n,
            "diminishing_returns": diminishing_returns,
            "search_path": search_path,
            "curve": curve,
            "n_trials_per_point": n_trials,
        }

    def cost_optimal_plan(self, true_p_before: float, effect_size: float,
                           cost_per_query: float, target_power: float = 0.80,
                           n_comparisons: int = 1, **kwargs) -> dict:
        """
        Wraps find_required_n() with a direct cost translation, and flags
        the concrete wasted spend if you tested at a larger n than needed.
        """
        result = self.find_required_n(true_p_before, effect_size, target_power,
                                       n_comparisons, **kwargs)
        if result["required_n"] is not None:
            result["required_cost"] = result["required_n"] * cost_per_query
        else:
            result["required_cost"] = None
        for row in result["curve"]:
            row["cost"] = row["sample_size"] * cost_per_query
        return result

    @staticmethod
    def to_dataframe(result: dict) -> pd.DataFrame:
        """
        Structured, non-printing alternative to print_search_report()'s
        curve table. One row per sample size tested during the search.
        """
        df = pd.DataFrame(result["curve"]).sort_values("sample_size").reset_index(drop=True)
        if result.get("required_n") is not None:
            df["is_required_n"] = df["sample_size"] == result["required_n"]
        return df

    def print_search_report(self, result: dict):
        print("=" * 70)
        print(f" SAMPLE SIZE SEARCH")
        print(f" True baseline: {result['true_p_before']:.0%}  |  "
              f"Target effect: +{result['effect_size']:.0%}  |  "
              f"n_comparisons: {result['n_comparisons']}  |  "
              f"Target power: {result['target_power']:.0%}")
        print("=" * 70)
        has_cost = "cost" in result["curve"][0] if result["curve"] else False
        header = f"    {'n':>6} {'detection rate':>16}"
        if has_cost:
            header += f" {'cost':>12}"
        print(header)
        for row in result["curve"]:
            marker = "  <-- required n" if row["sample_size"] == result["required_n"] else ""
            line = f"    {row['sample_size']:>6} {row['detection_rate']:>16.0%}"
            if has_cost:
                line += f" {row['cost']:>12,.2f}"
            print(line + marker)
        print("-" * 70)
        if result["required_n"] is not None:
            msg = f"  Minimum n for {result['target_power']:.0%} power: {result['required_n']}"
            if result.get("required_cost") is not None:
                msg += f"  (estimated cost: ${result['required_cost']:,.2f})"
            print(msg)
            print(f"  Testing at any n above this wastes budget without meaningfully")
            print(f"  improving your ability to detect this specific effect size.")
        else:
            if result.get("diminishing_returns"):
                last_n = result["search_path"][-1]["sample_size"]
                last_rate = result["search_path"][-1]["detection_rate"]
                print(f"  Stopped early at n={last_n} ({last_rate:.0%} power) -- doubling the sample")
                print(f"  size stopped meaningfully improving detection. This effect is likely too")
                print(f"  small relative to measurement noise to be worth chasing with more data.")
                print(f"  (Set diminishing_returns_threshold=0 to force the search to keep going to max_n.)")
            elif result.get("hit_max_n"):
                print(f"  Searched up to n={result['search_path'][-1]['sample_size']} and never reached "
                      f"{result['target_power']:.0%} power.")
                print(f"  This effect size may not be economically detectable at a sane sample size --")
                print(f"  consider a larger effect target, a lower target_power, or accept the cost of")
                print(f"  raising max_n further.")
            else:
                print(f"  None of the tested sample sizes reached {result['target_power']:.0%} power.")
                print(f"  Either this effect is too small to detect economically, or you")
                print(f"  need to test larger candidate sizes than were tried here.")
        print(f"\n  (Based on {result['n_trials_per_point']} simulated trials per sample size --")
        print(f"   increase n_trials for a more precise estimate before committing budget.)")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    np.random.seed(42)

    # --- TEST 1: TEAM A (Accuracy - "Higher is Better"), single metric, no correction ---
    before_acc = np.random.binomial(1, p=0.70, size=500)
    after_acc  = np.random.binomial(1, p=0.78, size=500)

    accuracy_auditor = AIBootstrapAuditor(metric_name="RAG LLM Accuracy", goal="increase")
    accuracy_report = accuracy_auditor.audit(before_acc, after_acc)
    accuracy_auditor.print_executive_report(accuracy_report)

    # --- TEST 2: TEAM B (Latency - "Lower is Better"), single metric, no correction ---
    before_lat = np.random.normal(loc=1200, scale=300, size=150)
    before_lat = np.clip(before_lat, 100, None)
    after_lat  = np.random.normal(loc=950, scale=250, size=150)
    after_lat  = np.clip(after_lat, 100, None)

    latency_auditor = AIBootstrapAuditor(metric_name="API Latency (ms)", goal="decrease")
    latency_report = latency_auditor.audit(before_lat, after_lat)
    latency_auditor.print_executive_report(latency_report)

    # --- TEST 3: Demonstrating the multiple-comparisons correction ---
    # Simulate a review cycle where 5 teams each test their own metric at once.
    # Same accuracy data as Team A, but now explicitly flagged as 1-of-5
    # simultaneous comparisons.
    print("### Same Team A result, but run as 1 of 5 simultaneous metrics in one review cycle: ###\n")
    accuracy_auditor_corrected = AIBootstrapAuditor(
        metric_name="RAG LLM Accuracy", goal="increase", n_comparisons=5
    )
    accuracy_report_corrected = accuracy_auditor_corrected.audit(before_acc, after_acc)
    accuracy_auditor_corrected.print_executive_report(accuracy_report_corrected)

    # --- TEST 4: AuditBatch -- the intended real-world usage ---
    # A review cycle with 5 metrics: the two above, plus 3 more where one is
    # deliberately borderline, to show the correction actually change a
    # verdict rather than just widening a CI that was never close to zero.
    print("### AuditBatch: full review cycle, correction applied automatically ###\n")

    np.random.seed(7)
    # A borderline metric: small true effect, small sample -- exactly the
    # kind of result that can look "significant" uncorrected but shouldn't
    # survive correction when it's one of several simultaneous tests.
    before_borderline = np.random.binomial(1, p=0.70, size=80)
    after_borderline = np.random.binomial(1, p=0.74, size=80)

    before_c = np.random.normal(loc=500, scale=80, size=200)
    after_c = np.random.normal(loc=470, scale=80, size=200)

    before_d = np.random.binomial(1, p=0.60, size=300)
    after_d = np.random.binomial(1, p=0.60, size=300)  # genuinely no effect

    batch = AuditBatch(confidence_level=0.95)
    batch.add_metric("RAG LLM Accuracy", before_acc, after_acc, goal="increase")
    batch.add_metric("API Latency (ms)", before_lat, after_lat, goal="decrease")
    batch.add_metric("Borderline Feature Flag", before_borderline, after_borderline, goal="increase")
    batch.add_metric("Support Response Time (s)", before_c, after_c, goal="decrease")
    batch.add_metric("Unrelated Control Metric", before_d, after_d, goal="increase")

    batch_results = batch.run(verbose=False)  # suppress individual reports, just show summary
    batch.print_batch_summary(batch_results)

    # --- TEST 5: GuardrailAuditor -- the "prompt got wordier" scenario ---
    # Prompt change makes accuracy genuinely better (core value metric),
    # but also makes responses genuinely longer and slightly slower
    # (guardrail metrics) -- exactly the tradeoff described in the prompt.
    print("### GuardrailAuditor: catching a real, moderate regression that a ###")
    print("### Bonferroni-style stricter test would likely have missed      ###\n")

    np.random.seed(11)
    before_tokens = np.random.normal(loc=180, scale=40, size=250)
    before_tokens = np.clip(before_tokens, 10, None)
    # A real, moderate increase (not huge) -- the "wordier answers" effect
    after_tokens = np.random.normal(loc=216, scale=45, size=250)
    after_tokens = np.clip(after_tokens, 10, None)

    token_guard = GuardrailAuditor(
        metric_name="Output Token Count",
        guard_direction="not_increase",
        margin=0,       # any real increase counts as a regression here
        alpha=0.05,
    )
    token_result = token_guard.audit(before_tokens, after_tokens)
    token_guard.print_executive_report(token_result)

    # A borderline/small-sample case, deliberately, to show INCONCLUSIVE
    # rather than a silent PASS when evidence is genuinely insufficient.
    np.random.seed(22)
    before_latency_small = np.random.normal(loc=800, scale=150, size=25)
    after_latency_small = np.random.normal(loc=830, scale=150, size=25)

    latency_guard = GuardrailAuditor(
        metric_name="Generation Latency p95 (ms, small sample)",
        guard_direction="not_increase",
        margin=20,
        alpha=0.05,
    )
    latency_result = latency_guard.audit(before_latency_small, after_latency_small)
    latency_guard.print_executive_report(latency_result)

    # --- TEST 6: The full diagram, wired together in one batch ---
    # Core value metrics -> Benjamini-Hochberg (find real quality wins)
    # Guardrail/cost metrics -> GuardrailAuditor (catch stealth regressions)
    # One unified report, two different statistical treatments underneath.
    print("### Full integration: core metrics (BH) + guardrails (GuardrailAuditor), one report ###\n")

    np.random.seed(99)
    before_int_acc = np.random.binomial(1, p=0.72, size=400)
    after_int_acc  = np.random.binomial(1, p=0.80, size=400)   # real accuracy win

    before_int_rel = np.random.normal(loc=0.70, scale=0.15, size=400)
    after_int_rel  = np.random.normal(loc=0.76, scale=0.15, size=400)  # real relevance win

    before_int_tokens = np.random.normal(loc=180, scale=40, size=400)
    before_int_tokens = np.clip(before_int_tokens, 10, None)
    after_int_tokens  = np.random.normal(loc=214, scale=42, size=400)  # real token regression
    after_int_tokens  = np.clip(after_int_tokens, 10, None)

    before_int_latency = np.random.normal(loc=900, scale=180, size=400)
    after_int_latency  = np.random.normal(loc=915, scale=180, size=400)  # small, tolerable drift

    integrated_batch = AuditBatch(confidence_level=0.95, correction_method="benjamini_hochberg")
    integrated_batch.add_metric("intent_classification_accuracy", before_int_acc, after_int_acc, goal="increase")
    integrated_batch.add_metric("response_relevance_llm_judge", before_int_rel, after_int_rel, goal="increase")
    integrated_batch.add_guardrail("output_token_count", before_int_tokens, after_int_tokens,
                                    guard_direction="not_increase", margin=0)
    integrated_batch.add_guardrail("generation_latency_p95", before_int_latency, after_int_latency,
                                    guard_direction="not_increase", margin=50)  # allow up to 50ms drift

    integrated_results = integrated_batch.run(verbose=False)
    integrated_batch.print_batch_summary(integrated_results)

    # --- TEST 7: Calibrating the harness itself ---
    print("### HarnessCalibrationSuite: validating the test's own long-run behavior ###\n")

    suite = HarnessCalibrationSuite(confidence_level=0.95, n_bootstrap_iterations=1000)

    null_result = suite.null_calibration(true_p=0.70, n_questions=200, n_trials=100, n_comparisons=1)
    power_rows = suite.power_curve(
        true_p_before=0.70,
        effect_sizes=(0.05, 0.08),
        sample_sizes=(200, 500),
        n_trials=40,
        n_comparisons=1,
    )
    coverage_result = suite.ci_coverage(true_p_before=0.70, true_effect=0.05, n_questions=200, n_trials=100)

    suite.print_full_report(null_result, power_rows, coverage_result)

    # --- TEST 8: Sample size optimizer -- checking the earlier unverified claims ---
    print("### SampleSizeOptimizer: computing (not assuming) required sample sizes ###\n")

    optimizer = SampleSizeOptimizer(confidence_level=0.95, n_bootstrap_iterations=800)

    # Claim to check: "n=150-200 is enough for a 15% swing-for-the-fences effect"
    result_big_effect = optimizer.find_required_n(
        true_p_before=0.70, effect_size=0.15, target_power=0.80,
        candidate_sizes=(50, 100, 150, 200, 300), n_trials=200,
    )
    optimizer.print_search_report(result_big_effect)

    # Claim to check: "n=1500+ needed for a 2-3% micro-optimization"
    # Previously (fixed grid up to 3000) this came back "None of the tested
    # sizes reached target power" -- unresolved. Adaptive search removes the
    # need to guess the range: it keeps doubling until it finds an answer
    # or hits max_n.
    result_small_effect = optimizer.find_required_n_adaptive(
        true_p_before=0.70, effect_size=0.03,
        target_power=0.80, start_n=500, max_n=50000, n_trials=200,
    )
    optimizer.print_search_report(result_small_effect)
    if result_small_effect["required_n"] is not None:
        cost = result_small_effect["required_n"] * 0.02  # 2 cents/LLM-judge call, example
        print(f"  At $0.02/query, that's an estimated ${cost:,.2f} to reliably detect this effect.\n")
