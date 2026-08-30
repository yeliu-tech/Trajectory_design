# Exp E — Model-family robustness of the M1-over-M0 benefit

Date: 2026-08-29. Script: `experiments/history_conditioned_ftg_final/scripts/sse_expE_model_robustness.py`.
Output dir: `experiments/state_sufficiency_extension/expE_model_robustness/`.

## 1. What was done

Question: is the incremental value of M1=M0+T over M0 confined to HGBR, or does it appear under learners with different inductive biases?

Three regression families, M0 (25D) vs M1 (35D) within each family, identical settings inside a family, no per-arm tuning:

| Family | Implementation | Notes |
|---|---|---|
| HGBR | frozen `models/exp21/model_m0/m1.joblib` | reused as-is, not retrained |
| ExtraTrees | `ExtraTreesRegressor(n_estimators=100, random_state=seed+i, n_jobs=-1)`, other params library default (sklearn 1.7.2) | newly trained, `models/sse/model_e_et_*.joblib` |
| MLP | `MLPRegressor(hidden_layer_sizes=(64,32), random_state=seed+i)`, other params library default; `StandardScaler` fit on each member's training subset only, validation only transformed — identical handling for M0/M1 | newly trained, `models/sse/model_e_mlp_*.joblib` |

All: same 5-member ensemble construction (member 0 full data; members 1–4 scenario-cluster bootstrap, `RandomState(seed+1000i)`, verbatim from `exp2_train_abc.train_ensemble`), same 988/9,597 train and 180/14,400 validation split, same J* target, same evaluation protocol (ensemble-mean prediction → cell metrics, scenario-cluster bootstrap B=1000, seed=2024).

## 2. Anchor

Recomputed HGBR M0/M1 metrics match the frozen exp31 on-disk values exactly (Δacc = Δregret = 0.00e+00, match=True; `anchor_check.json`).

## 3. Results (validation, 180 scenarios)

| Family | Arm | Pairwise acc [CI] | Mean top-1 regret [CI] | Median top-1 regret |
|---|---|---|---|---:|
| HGBR | M0 | 0.7933 [0.7809, 0.8054] | 12.13 [9.53, 15.33] | 0.425 |
| HGBR | M1 | 0.8193 [0.8077, 0.8302] | 8.53 [6.95, 10.17] | 0.205 |
| ExtraTrees | M0 | 0.8526 [0.8429, 0.8617] | 9.73 [7.72, 11.91] | 0.028 |
| ExtraTrees | M1 | 0.8643 [0.8567, 0.8719] | 8.68 [6.62, 10.88] | 0.000 |
| MLP | M0 | 0.7532 [0.7365, 0.7682] | 11.05 [9.23, 13.07] | 0.653 |
| MLP | M1 | 0.7763 [0.7633, 0.7898] | 13.38 [10.86, 16.28] | 0.439 |

Paired scenario-level contrasts, M1 vs M0 (positive = M1 better; `paired_contrasts.csv`):

| Family | Δ pairwise accuracy [95% CI] | Δ mean top-1 regret [95% CI] |
|---|---|---|
| HGBR | +0.0263 [+0.0209, +0.0320] | +3.59 [+1.58, +6.02] |
| ExtraTrees | +0.0115 [+0.0082, +0.0151] | +1.05 [−0.48, +2.53] |
| MLP | +0.0225 [+0.0152, +0.0304] | **−2.33 [−4.75, −0.29]** |

## 4. Interpretation (as observed, no cherry-picking)

1. **Pairwise ranking accuracy: direction is consistent across all three families.** M1 > M0 everywhere, and every 95% CI excludes zero. On this metric the augmentation benefit is **not confined to HGBR**.
2. **Mean top-1 regret: NOT consistent.** HGBR improves significantly; ExtraTrees improves in direction but the CI includes zero; **MLP reverses** — M1's mean regret is significantly *worse* (−2.33 [−4.75, −0.29]) even though its pairwise accuracy and median regret (0.439 vs 0.653) improve. The MLP M1 degradation is driven by the regret tail (mean up while median down), i.e. a small number of scenarios where the MLP misuses the extra inputs and picks a much worse top-1.
3. **MLP convergence caveat (material).** All 10 MLP members hit the library-default `max_iter=200` without converging (`tol` not met; losses fell ~300–500× but were still decreasing — `mlp_convergence_diagnostics.csv/.json`). The MLP arm is therefore an *undertrained* reference: the MLP mean-regret reversal is a real, significant result for this frozen setting, but it characterises an unconverged model and must not be generalised to "MLPs cannot exploit the augmented state". Correct scope: *the top-1 decision benefit is learner- and training-procedure-dependent.*
4. Therefore the correct claim is narrow: *the T-augmentation improves pairwise ranking across all three tested learners; the top-1 decision benefit is reproduced under both tree ensembles but not under the small (unconverged) MLP, where the mean-regret effect reverses despite better median regret and better pairwise accuracy.*
5. The pre-registered statement "the state augmentation benefit is not confined to HGBR" is supported **only for pairwise ranking accuracy**, and must not be extended to top-1 regret in the manuscript.

## 5. Manuscript-ready facts

1. Under ExtraTrees (100 trees, library defaults otherwise), M1 improves pairwise accuracy over M0 by +0.0115 [+0.0082, +0.0151]; under the small MLP (64×32), by +0.0225 [+0.0152, +0.0304] — same direction as HGBR (+0.0263 [+0.0209, +0.0320]).
2. For mean top-1 regret, HGBR improves (+3.59 [+1.58, +6.02]), ExtraTrees improves without resolving significance (+1.05 [−0.48, +2.53]), and MLP reverses (−2.33 [−4.75, −0.29]).
3. The MLP reversal is a tail effect: MLP median top-1 regret improves under M1 (0.653 → 0.439) while the mean worsens.
4. The augmentation's ranking benefit replicates across inductive biases; its top-1 decision benefit is learner-dependent.

## 6. Claim boundary

- Supported: pairwise-accuracy benefit generalises across three model families.
- NOT supported: "M1 benefit is not confined to HGBR" as a blanket statement (top-1 regret reverses under MLP); any claim that MLP/ExtraTrees are better surrogates than HGBR overall (ExtraTrees has the best raw metrics but was not tuned; this experiment is a robustness check, not a model competition); any strong conclusion from the MLP arm beyond this frozen setting (all 10 members stopped at max_iter=200 unconverged — see `mlp_convergence_diagnostics.json`).

## 7. Files

- `summary.csv`, `paired_contrasts.csv`, `per_scenario_metrics.csv`, `anchor_check.json`, `train_timing.json`
- `mlp_convergence_diagnostics.csv/.json` (per-member n_iter / loss curve summary)
- `figure_model_robustness.png/.pdf` (600 dpi)
- New models: `models/sse/model_e_et_m0/m1.joblib`, `model_e_mlp_m0/m1.joblib` (no frozen file touched)
