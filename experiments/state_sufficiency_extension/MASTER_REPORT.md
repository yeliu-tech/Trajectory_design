# MASTER REPORT — State-Sufficiency Extension Experiments (FINAL v2)

Status: **FINAL** — Exp A / B / C / D / E all complete. No manuscript text has been modified by this work; this report and `manuscript_implications.md` are the decision basis for the next manuscript revision.

Date: 2026-08-30. Root: `experiments/state_sufficiency_extension/`.
Frozen contract held throughout: train 988 scenarios / 9,597 rows, validation 180 scenarios / 14,400 rows, independent test 36 scenarios (untouched), J* production labels unchanged, HGBR 5-member / 300-iteration / seed-2024 contract unchanged, scenario-cluster bootstrap (B=1000, seed=2024) for all CIs. No frozen output overwritten; new models only in `models/sse/`; all negative/mixed results reported as observed.

---

## 1. Executive summary

- **Exp A** — The aliasing heavy tail is **not** an artefact of the 10D matching vector: matching on the full 25D M0 state reproduces the diagnosis almost exactly (6,605 pairs; median |ΔJ*|/σ = 0.0122 [0.0100, 0.0141]; P(>0.5σ) = 0.121 [0.112, 0.130]). "25D M0 aliasing" is defensible.
- **Exp B** — T decomposes into current8 + history2. Each alone gives a comparable significant gain over M0 (Δregret +3.02 / +2.98); full T is best on every metric. **Case C: task-relative reparameterization and limited path memory are complementary, partially redundant.** history2 alone (2 dims) recovers ~83% of the full-T regret gain.
- **Exp C** — On the diagnosed matched pairs, fullT converts representation separation into significantly better pair ordering vs M0 (+4.2 pts overall; +2.5 pts on high-aliasing pairs) and vs shuffle/random controls on every stratum. **No decision-level amplification in the severe tail** (ceiling effect; all models > 0.92 there).
- **Exp D** — At 8× completion budget (3200), the diagnosed aliasing **persists**: order agreement 98.5% [96.9, 100], 81.5% [75.7, 87.0] of high-aliasing pairs still exceed 0.5σ; matched low controls stay at 0.5%. Attenuation is confined to the extreme tail (28.5% retain >1.0σ): finite-budget noise inflates the largest gaps but does not explain the phenomenon. Determinism anchor vs exp26: 2/2 recomputed prefixes exact.
- **Exp E** — The M1-over-M0 **pairwise-accuracy** benefit replicates across HGBR / ExtraTrees / MLP (+0.0263 / +0.0115 / +0.0225, all CIs exclude zero). The **top-1 regret** benefit does not replicate under MLP (significant reversal −2.33 [−4.75, −0.29]; tail effect — MLP median regret improves). Robustness claims must be scoped to ranking accuracy.

## 2. Exp A — Full-25D M0 aliasing audit

Dir: `expA_full25_aliasing/`. Verbatim exp19 protocol, matching space extended 10D → 25D; anchor on 10D exact (6,593 pairs, match=True).

| Matching space | pairs | median |ΔJ*|/σ | P(>0.5σ) | P(>1.0σ) |
|---|---:|---|---|---|
| 10D original | 6,593 | 0.0125 | 0.1236 | 0.067 (10D ref) |
| 25D full M0 | 6,605 | 0.0122 [0.0100, 0.0141] | 0.1207 [0.1115, 0.1301] | 0.0671 |

Threshold sensitivity (0.25×/0.5×/1.0×): spaces track at every threshold. Caveat stated openly: `k_prefix` + 10 scenario-constant variables are constant in **all** 1,193 matching groups (verified column-by-column; each of the remaining 14 columns varies in ≥932/1,193 groups), so the effective matching dimension is **14** — a property of the state, not the protocol.

**Supports full-25D M0 aliasing: YES.** Figure 3 stands; add one sentence citing the 25D audit (Supplement table).

## 3. Exp B — T decomposition (current8 vs history2)

Dir: `expB_T_decomposition/`. New models `models/sse/model_b1/b2.joblib`; anchors B0/B3 exact vs frozen exp31.

| Arm | Dim | Pairwise acc [CI] | Mean top-1 regret [CI] |
|---|---:|---|---|
| B0 M0 | 25 | 0.7933 [0.7809, 0.8054] | 12.13 [9.53, 15.33] |
| B1 M0+current8 | 33 | 0.8100 [0.7980, 0.8209] | 9.11 [7.40, 10.85] |
| B2 M0+history2 | 27 | 0.8110 [0.7993, 0.8221] | 9.14 [7.36, 11.02] |
| B3 M0+fullT = M1 | 35 | 0.8193 [0.8077, 0.8302] | 8.53 [6.95, 10.17] |

Contrasts: B1−B0 +3.02 [+1.07,+5.35]; B2−B0 +2.98 [+1.19,+5.26]; B3−B0 +3.59 [+1.58,+6.02]; B3−B1 +0.58 [−0.51,+1.71] (acc +0.0097, sig); B3−B2 +0.61 [−0.36,+1.53] (acc +0.0086, sig).

**Case C — complementary.** Neither component dominates (rules out pure-reparameterization Case A and pure-history Case B); full T best on all metrics; components partially redundant (+3.59 < +3.02 + +2.98). Provenance detail (`expB_T_decomposition/current8_provenance.md`): current8 = 3 t2-relative + 5 t1-relative current-endpoint variables (endpoint functions only — re-parameterization), history2 = 2 t1-relative prefix-history functionals (running min ellipsoid distance / passed flag — genuine path memory, not reconstructible from the endpoint).

## 4. Exp C — Decision-level mechanism on diagnosed pairs

Dir: `expC_pair_decision/`. Six models × two pair populations (full-25D primary, orig-10D sensitivity) × three strata × two tie rules; out-of-sample predictions only.

full25D / noise-band: M0 0.6459 → fullT **0.6883** on all 5,425 pairs (paired Δ +0.0424 [+0.0266, +0.0560]); on 797 high-aliasing pairs 0.9335 → 0.9586 (Δ +0.0251 [+0.0075, +0.0453]); on 443 severe pairs Δ +0.0158 [−0.0042, +0.0350] (unresolved). fullT beats shuffle and random controls on every stratum.

**Conclusion**: representation separation converts into decision gains, broad rather than tail-concentrated. The manuscript must not claim decision-level amplification in the severe tail.

## 5. Exp D — High-budget J* relabel (budget 3200)

Dir: `expD_highbudget_relabel/`. 550 frozen pairs (200 high25 + 200 low25 controls + 150 sens10), 862 unique prefixes relabelled at 3200 with the frozen completion protocol (14 reused from exp26 by params hash; determinism anchor 2/2 exact, `anchor_recompute_check.json`).

| Set | n | Order agreement [CI] | Retain >0.5σ [CI] | Retain >1.0σ [CI] |
|---|---:|---|---|---|
| high25 | 200 | 0.985 [0.969, 1.000] | 0.815 [0.757, 0.870] | 0.285 [0.222, 0.351] |
| sens10 | 150 | 0.980 [0.954, 1.000] | 0.840 [0.775, 0.901] | 0.307 [0.233, 0.378] |
| low25 | 200 | 0.665 [0.600, 0.725] | 0.005 [0.000, 0.015] | 0.005 [0.000, 0.015] |

Prefix-level: Spearman(prod, 3200) = 0.980; median |drift| 2.55; 69.1% drift beyond the 0.35 noise band (labels refine, structure preserved).

**Verdict: the diagnosed aliasing persists at 8× budget** — ordering intact, 0.5σ heavy tail retained, high/low separation preserved; attenuation is confined to the >1σ extreme tail, which the manuscript should describe as noise-inflated rather than exact. 6000-budget confirmation was optional and not run; "labels converged" must not be claimed.

## 6. Exp E — Model-family robustness

Dir: `expE_model_robustness/`. HGBR anchors exact; ExtraTrees (100 trees) and MLP (64×32, train-only scaler); same ensemble/seed/data/protocol; no tuning.

| Family | Δacc (M1−M0) [CI] | Δregret (+ = M1 better) [CI] |
|---|---|---|
| HGBR | +0.0263 [+0.0209, +0.0320] | +3.59 [+1.58, +6.02] |
| ExtraTrees | +0.0115 [+0.0082, +0.0151] | +1.05 [−0.48, +2.53] |
| MLP | +0.0225 [+0.0152, +0.0304] | −2.33 [−4.75, −0.29] (median 0.653→0.439) |

**Conclusion**: ranking benefit is learner-robust; top-1 decision benefit is learner-dependent (MLP mean-regret reversal, tail-driven). Caveat: all 10 MLP members stopped at the library-default `max_iter=200` unconverged (`mlp_convergence_diagnostics.json`), so the MLP arm is an undertrained reference and the reversal must not be generalised beyond this frozen setting. Scope robustness claims to pairwise accuracy and report the MLP reversal explicitly.

## 7. Highest-level interpretation (task-book §7) — FINAL

**Interpretation 3: task-relative reparameterization + limited path history jointly contribute.**

Evidence chain: Exp A confirms the state-aliasing diagnosis is real in the full 25D space; Exp D confirms it is not a completion-budget artefact (ordering 98.5% preserved, 0.5σ tail retained); Exp B shows the correction's benefit comes from BOTH re-expressing endpoint geometry in target-relative coordinates AND adding two genuine prefix-history scalars (Case C, neither dominates); Exp C shows the correction converts into better decisions on the diagnosed pairs; Exp E shows the ranking benefit generalises across learners while the decision-level regret benefit is learner-dependent.

Interpretations 1 (pure missing path memory) and 2 (pure reparameterization) are each ruled out by Exp B. Interpretation 4 (evidence insufficient) is ruled out by Exp A + Exp D jointly.

## 8. Recommended figure/table updates — FINAL

- **Figure 3**: keep. Add one sentence: full-25D matched-pair audit reproduces the heavy tail (Supplement: Exp A table + threshold sensitivity). Add a second sentence: high-budget (3200) relabel preserves pair ordering (98.5%) and 0.5σ-tail retention (81.5%), while the >1σ extreme tail is partially noise-inflated (28.5% retention) — Supplement: Exp D figure + tables. Do not quote >1σ quantiles as exact in the main text.
- **Figure 4 / feature ladder**: unchanged. Add Exp B decomposition table to the **main text** (it materially sharpens the mechanism story); per-arm training details to Appendix.
- **Figure 5**: add Exp C decision-level results to Supplement (both populations, both tie rules); correct any main-text wording implying severe-tail decision amplification.
- **Model robustness**: Exp E table to Supplement; robustness sentence scoped to pairwise accuracy + explicit MLP reversal.
- **Nothing else changes**: data splits, J* protocol, ladder selection rule, terminal downstream results.

## 9. Reproducibility map

| Exp | Script | Key outputs | New models |
|---|---|---|---|
| A | `scripts/sse_expA_full25_aliasing.py` | `expA_full25_aliasing/` (pairs/summary/sensitivity/figure/report) | — |
| B | `scripts/sse_expB_T_decomposition.py` | `expB_T_decomposition/` (summary/contrasts/figure/report) | `models/sse/model_b1/b2.joblib` |
| C | `scripts/sse_expC_pair_decision.py` | `expC_pair_decision/` (accuracy/summary/figure/report) | — (uses frozen + Exp B) |
| D | `scripts/sse_expD_highbudget_relabel.py` (stages select/pilot/collect/analyze/anchor/figure) | `expD_highbudget_relabel/` (selected_pairs/unique_prefixes/Jstar_3200_raw/comparison/summary/meta/anchor/figure/report); checkpoint `I:/hcftg_scratch/sse_expD_highbudget/` | — |
| E | `scripts/sse_expE_model_robustness.py` | `expE_model_robustness/` (summary/contrasts/figure/report) | `models/sse/model_e_et_*/mlp_*.joblib` |

All scripts under `experiments/history_conditioned_ftg_final/scripts/`. All CIs: scenario-cluster bootstrap B=1000, seed=2024. All anchors passed (Exp A 10D exact; Exp B/E HGBR exact vs exp31; Exp D determinism 2/2 exact).
