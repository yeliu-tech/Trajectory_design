# Exp D — High-budget J* relabel of aliased pairs (budget 3200)

Date: 2026-08-30. Script: `experiments/history_conditioned_ftg_final/scripts/sse_expD_highbudget_relabel.py`.
Output dir: `experiments/state_sufficiency_extension/expD_highbudget_relabel/`.

## 1. What was done

Question: does the diagnosed aliasing heavy tail (Figure 3) reflect genuine state ambiguity, or is it an artefact of the finite 200→400 production completion budget?

- **Sample** (frozen selection, `selected_pairs.csv`, seed 20260829): 550 pairs = 200 high-aliasing full-25D pairs (`high25`, production |ΔJ*| > 0.5σ_s, from a pool of 797) + 200 matched low-aliasing controls (`low25`, < 0.25σ_s, scenario/depth-band matched) + 150 original-10D high-aliasing sensitivity pairs (`sens10`). 862 unique prefixes (`unique_prefixes.csv`). Stratified proportional allocation over (scenario, depth band); within-stratum rank-equispaced picks on |ΔJ*|. No model-error-based selection.
- **Relabel**: every unique prefix re-evaluated with the frozen completion-search implementation at budget 3200 (exp26 pattern: `orc.evaluate_prefix(exp1, sc, prefix, seed, *orc.budget_split(3200), review_fn, nsga_mode='nested')`, `content_seed(salt='hr_exp2')`). Objective, constraints, target acceptance and optimizer mixture unchanged. 14 prefixes already had 3200 results in exp26 and were reused by params_sha256; 848 were computed here (5 workers, checkpointed, crash-safe).
- **σ conventions**: primary normalisation uses the within-scenario production J* std (same as the diagnosis); sensitivity uses within-scenario 3200-label std where ≥5 selected prefixes exist (`dJ3200_over_std3200` columns).

## 2. Determinism anchor

exp26 reuse was validated by recomputing 2 of the 14 reused prefixes in this session: `anchor_recompute_check.json` (prefix 1: exp26 = recomputed = 125.87270657, exact). CRN determinism holds; reused values are interchangeable with fresh computation.

## 3. Results

Pair-level (`highbudget_summary.csv`; scenario-cluster bootstrap 95% CI, B=1000, seed=2024):

| Set | n pairs | n scen | Order agreement [CI] | Retain >0.25σ [CI] | Retain >0.5σ [CI] | Retain >1.0σ [CI] | Median |ΔJ*₃₂₀₀|/σ [CI] |
|---|---:|---:|---|---|---|---|---|
| high25 | 200 | 120 | **0.985 [0.969, 1.000]** | 0.925 [0.885, 0.964] | **0.815 [0.757, 0.870]** | 0.285 [0.222, 0.351] | 0.738 [0.683, 0.824] |
| sens10 (10D) | 150 | 94 | 0.980 [0.954, 1.000] | 0.913 [0.853, 0.960] | 0.840 [0.775, 0.901] | 0.307 [0.233, 0.378] | 0.480 (CI n/a) |
| low25 control | 200 | 100 | 0.665 [0.600, 0.725] | 0.005 [0.000, 0.015] | 0.005 [0.000, 0.015] | 0.005 [0.000, 0.015] | 0.0002 [0.0001, 0.0003] |

(`retain3200scale_*` columns give the σ₃₂₀₀-scaled sensitivity: high25 = 0.92 / 0.815 / 0.435 at 0.25/0.5/1.0σ.)

Prefix-level (`highbudget_summary_meta.json`): production vs 3200 labels correlate at Spearman ρ = 0.980 over 862 prefixes; median absolute drift 2.55; 69.1% of prefixes drift by more than the 0.35 noise band — individual labels refine substantially under 8× budget, but scenario-level structure is preserved.

## 4. Interpretation — does the aliasing persist?

**Verdict: the diagnosed aliasing largely persists under 8× completion budget, with partial attenuation confined to the extreme tail.**

1. **Ordering is essentially invariant**: 98.5% [96.9, 100] of production high-aliasing pairs keep the same J* ordering at budget 3200. The phenomenon cannot be ordering noise.
2. **The heavy tail persists at the diagnostic threshold**: 81.5% [75.7, 87.0] of production high-aliasing pairs still exceed 0.5σ at 3200 (92.5% exceed 0.25σ). The matched low-aliasing control stays at ~0.5% — the separation between diagnosed-high and diagnosed-low pairs is preserved, not washed out (Figure panel c).
3. **Partial attenuation at the extreme tail**: only 28.5% [22.2, 35.1] of pairs remain above 1.0σ (production scale). Finite-budget label noise inflates the *largest* production gaps, so the >1σ tail of Figure 3 should not be quoted as a precise quantity; the 0.25–0.5σ band is robust.
4. The original-10D sensitivity set (sens10) behaves identically (order agreement 0.980, retention 0.840 at 0.5σ), so the conclusion does not depend on the matching space.

Per the pre-registered three-way classification: **between "largely persists" and "partially attenuates but remains nontrivial" — we report it as: persists at the primary 0.5σ diagnostic threshold with ordering intact; attenuates at the >1σ extreme tail.**

## 5. Manuscript-ready facts

1. Relabelling 862 prefixes from the diagnosed matched pairs at 8× the production completion budget (3200 evaluations) preserves the pair ordering of production high-aliasing pairs in 98.5% [96.9, 100] of cases.
2. 81.5% [75.7, 87.0] of production high-aliasing pairs still exceed the 0.5σ diagnostic threshold under 3200-budget labels; matched low-aliasing controls remain below threshold at a rate of 0.5%.
3. Retention at the strictest 1.0σ cut drops to 28.5% [22.2, 35.1], indicating that finite-budget label noise inflates the extreme tail of the diagnosed gap distribution but does not explain the aliasing itself.
4. Prefix-level production and 3200-budget labels correlate at Spearman ρ = 0.980 despite 69% of individual labels drifting by more than the noise band.
5. Conclusion for the text: the aliasing heavy tail is a property of the state representation, not of the completion-search budget; the >1σ extreme quantiles should be described as inflated by label noise.

## 6. Claim boundary

- Supported: aliasing persists at 8× budget (ordering + 0.5σ retention); high/low separation preserved; extreme-tail inflation quantified.
- NOT supported: "labels have converged" (69% of prefixes still drift beyond the noise band; 6000-budget confirmation was not run — the task book made it optional); any claim about budgets beyond 3200.

## 7. Files

- `selected_pairs.csv`, `unique_prefixes.csv` (frozen sample), `Jstar_3200_raw.csv` (862 prefix labels), `highbudget_pair_comparison.csv` (550 pairs), `highbudget_summary.csv`, `highbudget_summary_meta.json`, `anchor_recompute_check.json`
- `figure_highbudget_aliasing.png/.pdf` (600 dpi)
- Checkpoint (resumable): `I:/hcftg_scratch/sse_expD_highbudget/sse_expD_j3200_checkpoint.csv`
