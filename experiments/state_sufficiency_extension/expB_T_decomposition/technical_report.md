# Exp B — T decomposition: current8 vs history2

Date: 2026-08-29. Script: `experiments/history_conditioned_ftg_final/scripts/sse_expB_T_decomposition.py`.
Output dir: `experiments/state_sufficiency_extension/expB_T_decomposition/`.

## 1. What was done

The frozen 10-variable target-approach group T was decomposed into:

- **current8** (8D, task-relative geometry at the current endpoint): bearing_rel_sin, bearing_rel_cos, off_target_angle_deg, off_target1_angle_deg, t1_dx, t1_dy, t1_dz, t1_dist
- **history2** (2D, prefix-history scalars): prefix_t1_min_ell, prefix_t1_passed

Four state arms were compared on the frozen validation population:

| Arm | State | Dim | Model |
|---|---|---:|---|
| B0 | M0 | 25 | frozen `models/exp21/model_m0.joblib` (reused) |
| B1 | M0+current8 | 33 | newly trained `models/sse/model_b1.joblib` |
| B2 | M0+history2 | 27 | newly trained `models/sse/model_b2.joblib` |
| B3 | M0+fullT = M1 | 35 | frozen `models/exp21/model_m1.joblib` (reused) |

All arms: same HGBR family, 5-member ensemble (member 0 full data, members 1–4 scenario-cluster bootstrap, seeds 2024+i), 300 boosting iterations, squared-error loss on continuous J*, same 988 train scenarios / 9,597 rows, same 180 validation scenarios / 14,400 rows, same evaluation protocol. Only the input columns differ.

## 2. Anchor verification

Recomputed B0 and B3 metrics match the frozen exp31 on-disk values to machine precision:

- B0 vs exp31 M0: Δacc = 0.00e+00, Δregret = 0.00e+00 → match=True
- B3 vs exp31 BT: Δacc = 0.00e+00, Δregret = 0.00e+00 → match=True

(`anchor_check.json`)

## 3. Results (validation, 180 scenarios; scenario-cluster bootstrap B=1000, seed=2024)

| Arm | Dim | Pairwise acc [CI] | Mean top-1 regret [CI] | Scenario-norm Spearman [CI] | Top-3 recall |
|---|---:|---|---|---|---:|
| B0 M0 | 25 | 0.7933 [0.7809, 0.8054] | 12.13 [9.53, 15.33] | 0.621 [0.593, 0.648] | 0.8485 |
| B1 M0+current8 | 33 | 0.8100 [0.7980, 0.8209] | 9.11 [7.40, 10.85] | 0.663 [0.640, 0.684] | 0.8535 |
| B2 M0+history2 | 27 | 0.8110 [0.7993, 0.8221] | 9.14 [7.36, 11.02] | 0.666 [0.642, 0.689] | 0.8560 |
| B3 M0+fullT | 35 | 0.8193 [0.8077, 0.8302] | 8.53 [6.95, 10.17] | 0.680 [0.657, 0.704] | 0.8629 |

Paired scenario-level contrasts (Δregret > 0 and Δacc > 0 favour the first-named arm; `paired_contrasts.csv`):

| Contrast | Δregret [95% CI] | Δaccuracy [95% CI] |
|---|---|---|
| B1 − B0 | +3.02 [+1.07, +5.35] | +0.0165 [+0.0107, +0.0223] |
| B2 − B0 | +2.98 [+1.19, +5.26] | +0.0177 [+0.0125, +0.0234] |
| B3 − B0 | +3.59 [+1.58, +6.02] | +0.0263 [+0.0209, +0.0320] |
| B3 − B1 | +0.58 [−0.51, +1.71] | +0.0097 [+0.0054, +0.0146] |
| B3 − B2 | +0.61 [−0.36, +1.53] | +0.0086 [+0.0049, +0.0126] |

## 4. Interpretation (per pre-registered case scheme)

**Case C — task-relative reparameterization and limited path memory are complementary.**

- current8 alone (B1) and history2 alone (B2) each deliver statistically significant, and nearly equal, gains over M0 (Δregret ≈ +3.0 both; Δacc ≈ +0.017 both). Neither component dominates: this rules out Case A (current8-only) and Case B (history2-only).
- fullT (B3) is the best arm on every metric. Its accuracy advantage over each single-component arm is significant (+0.0097 over B1, +0.0086 over B2, CIs exclude zero). Its mean-regret advantage over B1/B2 is positive (+0.58 / +0.61) but the 95% CIs include zero — the incremental regret benefit of combining both components is not statistically resolved at n=180 scenarios.
- The two components are therefore partially redundant (fullT gain over B0, +3.59, is less than the sum of the individual gains, +6.0), consistent with history2 being partially derivable from current8 geometry, but each still contributes independent ranking information.

Notably, history2 — two scalar prefix-summary variables — alone recovers ~83% of the full-T regret gain at only +2 dimensions, the most dimension-efficient single correction observed.

## 5. Manuscript-ready facts

1. Decomposing the frozen target-approach group into its 8 current task-relative variables (current8) and its 2 prefix-history scalars (history2), each subset alone improves over M0 by a comparable, statistically significant margin (Δregret +3.02 [+1.07, +5.35] and +2.98 [+1.19, +5.26]; Δaccuracy +0.0165 and +0.0177).
2. The full 10-variable correction (M1) remains the best arm on all metrics (accuracy 0.8193 [0.8077, 0.8302], mean top-1 regret 8.53 [6.95, 10.17]).
3. M1's pairwise-accuracy advantage over either component alone is significant (+0.0097 [+0.0054, +0.0146] over M0+current8; +0.0086 [+0.0049, +0.0126] over M0+history2), while the corresponding mean-regret differences are positive but not resolved (+0.58 [−0.51, +1.71]; +0.61 [−0.36, +1.53]).
4. The M1 gain over M0 (+3.59 regret) is smaller than the sum of the component gains (+6.0), indicating partial redundancy between current task-relative geometry and the two history scalars.
5. The two history scalars alone (27D total) achieve ~83% of the full-T regret reduction — the most dimension-efficient single correction in the ladder.

## 6. Claim boundary

- Supported: both current8 and history2 carry ranking-relevant information missing from M0; they are complementary; M1 (both) is the strongest arm and its accuracy edge over each component alone is significant.
- NOT supported: any claim that current8 or history2 alone explains the full M1 gain; any claim that the fullT-vs-component regret difference is significant (CIs include zero); any claim about terminal trajectory quality (Exp C/D address mechanism and label noise separately).

## 7. Files

- `summary.csv`, `paired_contrasts.csv`, `per_scenario_metrics.csv`, `global_metrics.json`, `anchor_check.json`, `train_timing.json`
- `figure_T_decomposition.png/.pdf` (600 dpi)
- New models: `experiments/history_conditioned_ftg_final/models/sse/model_b1.joblib`, `model_b2.joblib` (no frozen file touched)
