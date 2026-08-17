# Trajectory Design — Supplementary Experiments

Data, figures, and analysis scripts for two supplementary experiments accompanying the manuscript:

> **History-Conditioned Candidate Retention for Repairable Initialization in Well Trajectory Optimization**

The paper studies **state sufficiency** in sequential well-trajectory design: whether the state representation used by an AI-based candidate assessor contains enough task-relevant information to rank incomplete trajectory prefixes. The two experiments released here provide (A) computational-efficiency evidence for the learned assessor and (B) a post-selection specificity audit of the compact state correction.

## Repository layout

```
figures/                              # Main result figures (600 dpi PNG)
scripts/                              # Exact experiment scripts (see note on dependencies below)
experiments/
  exp30_efficiency/                   # Experiment A+B: candidate-assessment efficiency
    technical_report.md               #   Full report (audit, numbers, claim boundaries)
    efficiency_summary.json           #   Headline latency / speedup numbers
    candidate_timing_summary.csv      #   Per-method timing summary
    candidate_timing_raw.csv          #   Per-candidate raw timings
    budget_tradeoff_summary.csv       #   Retained-budget K vs ranking quality
    budget_tradeoff_raw.csv
    amortized_cost.csv                #   Amortized cost vs number of candidates
    subset_ids.csv                    #   Stratified benchmark subset (selection rule included)
    run_manifest.json                 #   Environment, seeds, frozen-input hashes
  exp31_targeted_specificity/         # Experiment C: targeted correction specificity audit
    technical_report.md
    mechanism_group_summary.csv       #   Per-group aliasing-pair separation + Spearman
    mechanism_paired_contrasts.csv    #   T vs G/R/S paired contrasts
    pure_arm_ranking_summary.csv      #   M0 / M0+G / M0+R / M0+T / M0+S / full-history ranking
    ranking_paired_BT_vs_alternatives.csv
    burden_specificity_summary.csv    #   Gain vs M0 by aliasing-burden stratum
    table_burden_slopes.csv           #   Robust burden-slope regressions
    table_T_vs_alternatives_high_burden.csv
    performance_complexity.csv        #   Performance–complexity frontier data
    audit_manifest.json               #   Frozen-protocol anchors and reused modules
    group_feature_lists.json          #   Exact G / R / T / S feature definitions
```

## Background

An AI-based candidate assessor (a frozen five-member histogram-based gradient-boosting regression ensemble, HGBR) predicts the completion-to-go quality `J*` of an incomplete trajectory prefix and ranks candidates for retention. Two state representations are compared under an identical architecture and training protocol:

- **M0** — 25-dimensional conventional checkpoint state;
- **M1** — M0 + a 10-dimensional compact target-approach correction (35D total).

## Experiment A — Candidate-assessment computational efficiency

On 270 held-out prefixes from 18 validation scenarios (same machine, same benchmark session):

| Method | Median time / candidate | Throughput |
|---|---|---|
| Direct completion search (production protocol) | 20.41 s | ~0.05 candidates/s |
| HGBR surrogate, M1 (feature extraction + 5-member inference) | 24.92 ms | ~20,700 candidates/s (batch) |
| HGBR surrogate, M0 | 24.94 ms | — |

- Median per-candidate speedup: **790×** (scenario-cluster bootstrap 95% CI 735–854; geometric mean 825×, CI 757–899).
- One-time offline training of the frozen M1 ensemble: **2.72 s** (9,597 rows, 988 scenarios) — break-even below a single direct evaluation.
- Extending the state from 25D to 35D changes online cost by <0.1%.

![Efficiency](figures/figure_efficiency_600dpi.png)

## Experiment B — Retained-candidate budget vs ranking quality

Top-K retention from the frozen validation pools (K = 1, 2, 3, 4, 6, 8):

- At K = 1, M1 significantly lowers retained-set regret vs M0 (Δ = −3.01, 95% CI [−6.03, −0.31]).
- True-best survival@K does not differ significantly at any K; **a reduction in retained-candidate budget is not demonstrated** (reported as a boundary result).

![Budget trade-off](figures/figure_budget_regret_600dpi.png)

## Experiment C — Targeted correction specificity audit (post-selection)

The target-approach correction T (frozen by an earlier performance–complexity selection rule) is audited against the other engineering feature groups — geometry evolution (G, 26D), design-resource accounting (R, 5D), safety-corridor evolution (S, 4D) — on the identical population of 6,593 M0-matched prefix pairs:

- **Mechanism correspondence:** T is *not* the group most strongly associated with the diagnosed aliasing — R shows the highest distance↔|ΔJ*| association (median Spearman ρ = 0.375 vs 0.318 for T; paired difference −0.062, 95% CI [−0.081, −0.042]). This negative result is reported as-is.
- **Ranking (identical HGBR, only state columns differ):** M0+T achieves the lowest mean top-1 regret (8.53) and highest pairwise accuracy (0.819); paired improvements over **every** alternative — including the 70D full-history arm (Δregret +3.27, 95% CI [+1.59, +5.04]) — exclude zero.
- **Severity specificity:** only M0+T has a significantly positive aliasing-burden slope (+11.91, 95% CI [+2.90, +20.92], p = 0.0095); in the high-burden stratum its gain over M0 (+5.90, 95% CI [+2.45, +9.90]) significantly exceeds all alternatives.
- **Economy:** M0+T lies on the performance–complexity Pareto frontier and strictly dominates the 70D full-history arm at half the dimension. Adding the 26D geometry group *degrades* ranking below M0.

![Specificity audit](figures/figure_targeted_correction_600dpi.png)

## Reproducibility notes

- All confidence intervals use the **scenario** as the resampling unit (scenario-cluster bootstrap, B = 1000, seed 2024); candidate-level rows are never treated as independent samples.
- The scripts in `scripts/` are the exact experiment drivers. They import the project's frozen internal modules (feature extraction, completion oracle, HGBR training), which are not part of this release; the scripts are provided for protocol transparency rather than standalone execution. All reported numbers can instead be audited directly from the CSV/JSON files in `experiments/`.
- Nothing in the frozen protocol (M1 selection rule, train/validation split, labels, matched pairs, HGBR hyperparameters) was modified for these experiments; see `audit_manifest.json` / `run_manifest.json` for anchors and input hashes.
- Trained model binaries (~17 MB, joblib) are not included; the summary statistics they produced are fully tabulated in the CSVs.

## Claim boundaries

These experiments support statements about **candidate assessment** only:

- The HGBR surrogate makes completion-to-go assessment ~3 orders of magnitude cheaper — not "the whole optimizer is X× faster";
- M1 improves ranking quality with negligible extra online cost — not "total optimization cost is reduced by X%" (downstream evaluator budgets were not measured);
- M0+T is the best among the *tested* engineering corrections — not a universally optimal or minimal sufficient state.

## License

To be determined by the authors.
