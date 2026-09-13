# Exp F — Independent Ranking Confirmation (M0 vs M1) — Technical Report

## A. One-sentence verdict

On the frozen 36-scenario independent population (planner decision units, never used for diagnosis/selection/training), the frozen M1 model improves over frozen M0 in the **same direction as validation on both pre-registered primary metrics** — pairwise accuracy +0.0027 and mean top-1 regret −69.70 — so the pre-registered direction check is **confirmed**, but the accuracy gain is small with a CI crossing zero, and the regret gain is carried mainly by the strong-path-dependence tail (median paired Δregret = 0).

## B. What was done (audit trail)

- **Population**: exp18 frozen decision units rebuilt verbatim via `exp18_sdftg_independent_check.build_units` — 1,005 units over 36 independent scenarios (five planner arms; ~27.9 units/scenario, ~7.1 candidates/unit). No unit definition change.
- **Labels**: `exp6_labels_checkpoint.csv` `J_star_cont` (production protocol; not recomputed).
- **Features**: exp18 content-addressed checkpoint (4,394 rows); runtime assertion confirmed **100% candidate coverage, 0 recomputation**.
- **Models**: frozen `models/exp21/model_m0.joblib` (25D) and `model_m1.joblib` (35D), score = 5-member ensemble mean μ (validation ranking convention, lower is better). **No training, no label regeneration, no split change.**
- **Statistics**: scenario is the unit; cluster bootstrap B=1000, seed=2024 (same as exp18). Pairwise accuracy pooled per scenario (exp2 convention: pairs with |ΔJ*|≥1e-9, 0.5 credit for exact prediction ties); regret/top3/CFO averaged unit→scenario.
- Input SHA-256 hashes recorded in `run_manifest.json`.

## C. Pre-registered decision rule (registered in the script docstring before the run)

- **confirmed** if Δpairwise_accuracy = acc(M1)−acc(M0) > 0 **and** Δtop1_regret = regret(M1)−regret(M0) < 0 (point-estimate direction match with validation);
- otherwise report mixed / not confirmed as-is;
- CIs are descriptive, not significance gates; magnitudes are descriptive only (planner decision-unit population ≠ validation uniform cells; no equivalence test against val).

## D. Results

Primary metrics (36 scenarios):

| metric | M0 | M1 | Δ(M1−M0) | 95% CI of Δ |
|---|---:|---:|---:|---|
| pairwise accuracy | 0.6113 | 0.6159 | **+0.0027** | [−0.0118, +0.0157] |
| mean top-1 regret | 93.22 | 23.52 | **−69.70** | [−193.72, −1.33] |

Secondary (continuity with exp18): top3_recall 0.6295→0.6329 (Δ+0.0034, CI [−0.0152, +0.0225]); CFO 0.3648→0.3494 (Δ−0.0154, CI [−0.0788, +0.0448]).

**Verdict: confirmed** (both primary point estimates in the validation direction).

Reference (different population, descriptive only — val180 uniform cells, expB): pairwise accuracy 0.7933→0.8193 (Δ+0.0260); mean top-1 regret 12.13→8.53 (Δ−3.59). Direction consistent on both.

## E. Honest caveats (must accompany any use of this result)

1. **Effect size is much smaller than validation on pairwise accuracy** (+0.0027 vs +0.0260), and its CI crosses zero. The planner decision-unit population is harder (base accuracy 0.61 vs 0.79) and structurally different; the two numbers are not comparable as magnitudes.
2. **The regret improvement is tail-driven.** Scenario-level regret is extremely skewed (M0: mean 93.2, median 2.0, p90 16.5, max 2949.8). The paired median Δregret is **0.00** and only 50% of scenarios improve; the mean gain comes from a small number of high-regret scenarios, mostly in the strong-path-dependence stratum.
3. **Stratum description (descriptive, not gated)**:
   - strong: regret 241.1→57.3, CFO 0.386→0.311, top3 0.636→0.650 — all M1-better;
   - simple: regret 7.83→2.12, CFO 0.401→0.320, top3 0.619→0.631 — all M1-better;
   - **medium: regret 3.53→4.82, CFO 0.302→0.415, top3 0.630→0.620 — all M1-worse (descriptive reversal).**
4. Regret values here are on the planner-pool J* scale (pools contain very poor candidates), so absolute regret is not comparable to validation cell regret.
5. This experiment confirms **ranking direction on an independent population**; it does not change the terminal equal-cost null result, and it does not validate repairability claims.

## F. Interpretation boundary

Supported:
- The M0→M1 ranking improvement direction replicates on a fully independent scenario population under frozen models, frozen labels, and the pre-registered rule.
- The benefit profile is consistent with the main-text claim that gains concentrate where path dependence is strong (strong stratum improves on all recorded metrics).

Not supported / must not be claimed:
- A large or statistically resolved independent accuracy gain (CI crosses zero).
- Uniform per-scenario improvement (median paired Δregret = 0; medium stratum descriptively reverses).
- Any terminal/downstream quality improvement; any learner-independent claim; any repairability claim.

## G. Files

- `unit_metrics_raw.csv` — 2,010 unit×arm rows (1,005 units × M0/M1)
- `scenario_metrics.csv` — 36 scenario-level rows
- `paired_contrast.csv` — primary/secondary metrics with bootstrap CIs
- `val_comparison.csv` — independent vs val side-by-side (different populations, descriptive)
- `run_manifest.json` — decision rule, verdict, input hashes, stratum means, timing
- Script: `experiments/history_conditioned_ftg_final/scripts/sse_expF_independent_confirmation.py`
