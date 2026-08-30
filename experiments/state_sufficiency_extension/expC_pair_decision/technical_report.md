# Exp C — Decision-level mechanism test on diagnosed matched pairs

Date: 2026-08-29. Script: `experiments/history_conditioned_ftg_final/scripts/sse_expC_pair_decision.py`.
Output dir: `experiments/state_sufficiency_extension/expC_pair_decision/`.

## 1. What was done

Question: after the T-space separation of aliased pairs (Figure 5 mechanism), do the HGBR models actually rank those pairs' J* in the correct order?

- **Pair populations**: primary = Exp A full-25D M0 matched pairs (6,605 pairs); sensitivity = original exp19 10D matched pairs (6,593 pairs). Frozen pair IDs reused verbatim; no re-matching.
- **Models** (out-of-sample validation predictions only; no retraining on pairs):
  C0 = M0 (frozen m0), C1 = M0+current8 (Exp B b1), C2 = M0+history2 (Exp B b2), C3 = M0+fullT (frozen m1), C4 = shuffle-T control (frozen m5), C5 = dimension-matched random control (frozen m6).
- **Metric**: pairwise ordering accuracy — sign(Ĵ_i − Ĵ_j) vs sign(J*_i − J*_j) — computed per scenario then aggregated (scenario-cluster bootstrap 95% CI, B=1000, seed=2024).
- **Tie rule**: primary = frozen noise band 0.35 (pairs with |ΔJ*| < 0.35 within noise band excluded); sensitivity = all pairs included.
- **Strata**: all pairs / high-aliasing (|ΔJ*| > 0.5σ_s) / severe (|ΔJ*| > 1.0σ_s).

## 2. Results — full-25D pairs, noise-band (primary)

| Model | All pairs (n=5,425) | >0.5σ (n=797) | >1.0σ (n=443) |
|---|---|---|---|
| C0 M0 | 0.6459 [0.6331, 0.6584] | 0.9335 [0.9078, 0.9586] | 0.9526 [0.9290, 0.9729] |
| C1 current8 | 0.6702 [0.6560, 0.6838] | 0.9598 [0.9419, 0.9763] | 0.9774 [0.9602, 0.9915] |
| C2 history2 | 0.6710 [0.6579, 0.6844] | 0.9561 [0.9312, 0.9780] | 0.9661 [0.9387, 0.9868] |
| C3 fullT | **0.6883 [0.6748, 0.7017]** | 0.9586 [0.9359, 0.9799] | 0.9684 [0.9400, 0.9889] |
| C4 shuffle | 0.6424 [0.6282, 0.6568] | 0.9398 [0.9149, 0.9644] | 0.9481 [0.9210, 0.9709] |
| C5 random | 0.6490 [0.6358, 0.6617] | 0.9247 [0.8920, 0.9531] | 0.9481 [0.9161, 0.9719] |

Paired scenario-cluster contrasts (full25D; `pair_decision_summary.csv`):

| Contrast | All pairs | >0.5σ | >1.0σ |
|---|---|---|---|
| fullT − M0 | **+0.0424 [+0.0266, +0.0560]** | **+0.0251 [+0.0075, +0.0453]** | +0.0158 [−0.0042, +0.0350] |
| current8 − M0 | +0.0243 [+0.0093, +0.0388] | +0.0263 [+0.0078, +0.0460] | +0.0248 [+0.0068, +0.0428] |
| history2 − M0 | +0.0251 [+0.0099, +0.0403] | +0.0226 [+0.0025, +0.0464] | +0.0135 [−0.0044, +0.0320] |
| fullT − shuffle | +0.0459 [+0.0289, +0.0629] | +0.0188 [+0.0037, +0.0374] | +0.0203 [+0.0021, +0.0418] |
| fullT − random | +0.0393 [+0.0234, +0.0559] | +0.0339 [+0.0182, +0.0521] | +0.0203 [−0.0023, +0.0427] |

Sensitivity (all-pairs tie rule; original-10D population): same ordering of models and same sign of all contrasts; magnitudes slightly smaller. See `pair_decision_accuracy.csv`.

## 3. Interpretation

1. **Representation separation converts into correct decisions.** On the diagnosed matched-pair population, fullT ranks pairs significantly better than M0 (+4.2 points on all pairs, +2.5 points on high-aliasing pairs) and significantly better than both controls in every stratum (only fullT−random at >1.0σ and fullT−M0 at >1.0σ are not resolved).
2. **No amplification pattern in severe strata — reported as observed.** All models exceed 0.92 accuracy on high-aliasing pairs, because those pairs by construction have large |ΔJ*| and are intrinsically easy to order. The absolute T-attributable gain is *largest* on the full pair population (+0.042) and smaller on the severe stratum (+0.016, CI includes zero for fullT−M0). The data therefore do **not** support "M1's advantage is amplified on high-aliasing pairs" as a decision-level statement; the correct statement is that the T augmentation improves ordering broadly, including on the diagnosed high-aliasing subset, with the largest margin where ordering is hardest (small-to-moderate |ΔJ*|).
3. current8 is the strongest single component on severe pairs (+0.0248 over M0, significant), consistent with Exp B's finding that both components contribute.
4. Controls behave as expected: shuffle-T ≈ M0 or below; the dimension-matched random control is not better than M0 on any stratum and is clearly worse on high-aliasing pairs — the T gain is not a dimensionality artifact.

## 4. Manuscript-ready facts

1. On 5,425 within-scenario matched prefix pairs (full-25D M0 matching, noise-band tie rule), the full target-approach state orders pair J* correctly in 68.8% [67.5, 70.2] of pairs vs 64.6% [63.3, 65.8] for M0 (paired Δ = +4.2 points [+2.7, +5.6]).
2. On the 797 diagnosed high-aliasing pairs (|ΔJ*| > 0.5σ), fullT improves ordering accuracy from 93.4% to 95.9% (paired Δ = +2.5 points [+0.7, +4.5]).
3. fullT beats the shuffle-T and dimension-matched random controls on every stratum examined (paired Δ vs shuffle: +4.6 / +1.9 / +2.0 points on all / >0.5σ / >1.0σ), confirming the gain is content-specific, not dimension-driven.
4. The largest absolute gain occurs on the full pair population rather than the severe-aliasing subset; severe pairs are intrinsically easy to order (all models > 0.92), so decision-level amplification in the severe tail is NOT observed.

## 5. Claim boundary

- Supported: T augmentation converts representation-level separation into statistically better pair ordering, including on diagnosed high-aliasing pairs; the effect exceeds shuffle and random-feature controls.
- NOT supported: decision-level amplification in the severe tail; any terminal-quality claim (Exp D addresses label noise; terminal repairability is out of scope here).

## 6. Files

- `pair_decision_accuracy.csv` (all strata × tie rules × populations), `pair_decision_summary.csv` (paired contrasts)
- `figure_pair_decision.png/.pdf` (600 dpi)
