# Manuscript Implications — State-Sufficiency Extension (FINAL)

Date: 2026-08-30. Based on completed Exp A/B/C/D/E. No manuscript text has been modified by this work.

## What the new evidence changes

### 1. Figure 3 / aliasing claim — strengthened, keep as is (Exp A)

The heavy tail survives matching on the **full 25D M0** state (median |ΔJ*|/σ = 0.0122, P(>0.5σ) = 0.121 — statistically indistinguishable from the 10D audit). The manuscript may keep the strong "checkpoint-state aliasing" wording and should add one sentence: the diagnosis is unchanged when matching uses the complete 25-variable checkpoint state (Supplement table from `expA_full25_aliasing/`). No figure replacement needed.

### 2. Mechanism story — sharpen to "complementary" (Exp B)

The decomposition shows current8 (task-relative re-expression) and history2 (two prefix-history scalars) each deliver comparable significant gains (Δregret ≈ +3.0 each), with the full T best on all metrics and partially redundant (+3.59 < sum). Recommended manuscript framing:

> The benefit of the target-approach group is not attributable to a single mechanism: re-expressing endpoint geometry in target-relative coordinates and adding two prefix-history scalars each recover a comparable share of the gain, and combining them yields the best ranking on every metric. The two information sources are partially redundant but complementary.

This supports **Interpretation 3** (reparameterization + limited path memory jointly contribute), not a pure "missing path memory" story. The decomposition table belongs in the main text; per-arm training details in the Appendix.

### 3. Figure 5 / decision-level mechanism — add, with a correction (Exp C)

New evidence: on the diagnosed matched pairs, fullT improves pair ordering over M0 (+4.2 points on all pairs; +2.5 points on high-aliasing pairs) and over shuffle/random controls on every stratum. **However**, the largest gain is on the full pair population, and severe-tail amplification is NOT observed (all models > 0.92 on >1.0σ pairs). Any current or planned text claiming that M1's advantage concentrates on high-aliasing pairs *at the decision level* must be rewritten as: the aliasing-burden amplification holds for the *regression/ranking metrics over the full validation population* (exp29/exp31), while on diagnosed pairs the decision-level gain is broad rather than tail-concentrated.

### 4. Model robustness — scope the claim (Exp E)

Safe sentence: "The ranking benefit of the augmented state replicates across three regression families (HGBR, ExtraTrees, MLP; Δ pairwise accuracy +0.0263 / +0.0115 / +0.0225, all 95% CIs excluding zero)."

Mandatory honesty: "The mean top-1 regret improvement is reproduced under both tree ensembles but reverses under the small MLP (−2.33 [−4.75, −0.29]), where the median regret nonetheless improves — the top-1 decision benefit is therefore learner-dependent, driven by tail behaviour." Do not write "not confined to HGBR" without this qualification. Additional caveat: all 10 MLP members stopped at the library-default max_iter=200 unconverged (`mlp_convergence_diagnostics.json`); the reversal characterises an undertrained reference and must not be generalised to MLPs as a class.

### 5. Label-noise robustness of the diagnosis — RESOLVED (Exp D: persists)

Exp D relabelled 862 prefixes from the diagnosed pairs at 8× budget (3200). Outcome: **the aliasing persists** — order agreement 98.5% [96.9, 100]; 81.5% [75.7, 87.0] of high-aliasing pairs still exceed 0.5σ; matched low controls remain at 0.5%. Attenuation is confined to the >1σ extreme tail (28.5% retention). Required manuscript actions:

- Main text, §diagnosis: add one sentence — "Recomputing completion-to-go labels for the diagnosed pairs at 8× the production budget preserves pair ordering in 98.5% of cases and retains the heavy tail at the 0.5σ threshold (81.5% retention), while matched low-aliasing controls remain below threshold (0.5%); the aliasing is therefore a property of the state representation, not of the completion-search budget."
- Limitations: state that >1σ extreme-tail quantiles are partially inflated by finite-budget label noise (28.5% retention) and should not be quoted as exact.
- Supplement: Exp D figure + pair/prefix tables. Do NOT claim "labels converged" (69% of prefixes still drift beyond the noise band; 6000-budget check not run).

## Suggested next-version structure deltas

- Main text: + Exp B decomposition table (§feature ladder), + Exp A 25D sentence (§diagnosis), + Exp D persistence sentence (§diagnosis) + extreme-tail caveat (§limitations), corrected Exp C framing (§mechanism).
- Supplement: Exp A full-25D table + sensitivity, Exp C pair-decision tables (both populations, both tie rules), Exp D figure + tables, Exp E robustness table.
- No change to: data splits, J* protocol description, feature ladder selection rule, terminal downstream results.
