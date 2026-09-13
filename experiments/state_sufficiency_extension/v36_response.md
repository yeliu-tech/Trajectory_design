# v36 Remaining Issues — Response Document

Date: 2026-08-30. Scope: answers to `v36_revision_and_remaining_issues.md` §3 (P1–P3).
Every claim below cites a landed artifact. No manuscript text was changed.

---

## P1a — No independent confirmation of the M0→M1 ranking effect

**Status: addressed by a new experiment (Exp F).**

A frozen, untouched confirmation population did exist: the 36 independent
scenarios' planner decision units, previously used only for the downstream
equal-cost test. We ran the frozen M0/M1 models on it — prediction and
statistics only; no training, no relabelling, no split change, and no
post-hoc splitting of the 180 validation scenarios (which the review
correctly prohibits).

- Experiment: `expF_independent_confirmation/` (script
  `experiments/history_conditioned_ftg_final/scripts/sse_expF_independent_confirmation.py`).
- Population: 1,005 exp18 frozen decision units over 36 scenarios
  (~27.9 units/scenario, ~7.1 candidates/unit); feature coverage 100% from
  the existing exp18 checkpoint (0 recomputation).
- Pre-registered rule (in the script docstring before the run): confirmed
  iff Δpairwise_accuracy > 0 and Δtop1_regret < 0 in point estimates.

**Result: confirmed, with material caveats.**

| metric | M0 | M1 | Δ(M1−M0) | 95% CI |
|---|---:|---:|---:|---|
| pairwise accuracy | 0.6113 | 0.6159 | +0.0027 | [−0.0118, +0.0157] |
| mean top-1 regret | 93.22 | 23.52 | −69.70 | [−193.72, −1.33] |

Caveats that must travel with this result (full list in
`expF_independent_confirmation/technical_report.md` §E):

1. The accuracy gain is small and its CI crosses zero (val gain was
   +0.0260 on a different, uniform-cell population; magnitudes are not
   comparable).
2. The regret gain is tail-driven: median paired Δregret = 0.00, 50% of
   scenarios improve; the mean is carried by strong-path-dependence
   scenarios (strong stratum regret 241.1→57.3, all metrics M1-better).
3. The medium stratum descriptively reverses on regret/top3/CFO.
4. This confirms ranking direction only; it does not touch the terminal
   equal-cost null result.

**Manuscript consequence:** the "independent ranking confirmation" open
question listed in v36 can now be closed, but it should be reported with
exactly the caveats above — the honest summary is "direction replicates
independently; the benefit concentrates in strong-path-dependence
scenarios; the independent accuracy effect is small."

## P1b — Target-acceptance convention remains physically awkward

**Status: recommendation adopted; no recomputation.**

We agree with the review's submission implication and will not regenerate
any labels to improve the representative case. Concrete handling for the
next revision:

1. Apply a fixed, non-cherry-picked selection rule over the existing 36
   independent terminal results, e.g.: among scenarios with joint pass
   under the registered tolerance, take the first (by frozen scenario
   order) whose refined target-ellipsoid norm² ≤ 1.0 (nominal boundary).
2. If no scenario satisfies the nominal boundary, move the representative
   trajectory case to the Supplement, keep the registered-tolerance case
   there with the actual threshold (4.3533) stated in the caption, and
   remove the case from the main text rather than re-running anything.
3. Keep the v36 Limitations statement that the registered tolerance is
   internally consistent but not dimensionally equivalent to the nominal
   d ≤ 1 boundary.

This is a presentation change only; no experiment is required or
performed.

## P2a — Pure G/R/T/S specificity audit is post-selection

**Status: narrative discipline already in place.**

All extension reports state the chronology explicitly: the original
selection was the pre-registered cumulative performance–complexity
ladder; the pure-arm comparison is a **post-selection specificity
audit** of the already-frozen M1. See `MASTER_REPORT.md` and
`exp_targeted_correction_specificity`-era wording carried into
`experiments/state_sufficiency_extension/MASTER_REPORT.md` (selection
logic section). The abstract/results must never imply a pre-registered
four-way competition — this is a wording rule for the next revision, no
new evidence needed.

## P2b — current8 weakens a pure "history" story → provenance audit

**Status: provenance audit completed before this review round.**

`expB_T_decomposition/current8_provenance.md` (verified against
`config/seg_state_features.py`, not inferred from names):

- **3 t2-relative variables** (`bearing_rel_sin/cos`,
  `off_target_angle_deg`): deterministic re-expressions of quantities M0
  already contains (inc, azi_sin/cos, tgt_dx/dy/dz) → **representation
  engineering**.
- **5 t1-relative current variables** (`off_target1_angle_deg`,
  `t1_dx/dy/dz`, `t1_dist`): M0's 25 columns contain only `t1_radius_h`,
  **not** the t1 coordinates — so these are **information addition**
  (genuinely new geometric content), not re-expression.
- **history2** (`prefix_t1_min_ell`, `prefix_t1_passed`): path functionals
  over all prefix points; cannot be reconstructed from the endpoint pose
  → genuine path-history information.

Combined with the Exp B decomposition (current8 ≈ history2 individually,
full T best on accuracy), the defensible framing is exactly the one the
review recommends: **"task-aligned current geometry + limited path
memory"** — not "history is the missing state variable." The response to
"information addition vs representation engineering" is: **both**,
split 5/3 within current8 as above.

## P2c — Model-family robustness is mixed

**Status: accepted; no new model experiments.**

HGBR remains the explicitly fixed operational assessor. The Exp E
evidence is reported as sensitivity only:

- Pairwise-accuracy direction (M1 > M0) holds for HGBR, ExtraTrees, and
  the tested MLP;
- top-1 regret improves for HGBR and ExtraTrees but reverses under the
  fixed MLP procedure, whose members all hit the 200-iteration cap
  without converging (`expE_model_robustness/` diagnostics) — a
  training-procedure artifact, not a clean neural negative result.

Per the review's submission implication: no learner-independent claim
will be made; the manuscript should state cross-family support for the
ordering direction only.

## P3a — J* remains a finite-protocol target

**Status: no action.**

Agreed — the 3200-evaluation audit covers the primary aliasing claim
(98.5% ordering agreement; 81.5% of high pairs retain the 0.5σ
classification) and the limitation is already stated. Global convergence
over all 14,400 validation prefixes is not claimed.

## P3b — 790× is an assessment-stage speedup, not total optimization speedup

**Status: boundary enforced in all extension text.**

The number is used only as "candidate-assessment-stage speedup after
offline supervision." It will not appear as "faster optimization" in
title, highlights, cover letter, or graphical abstract. Offline label
generation cost and candidate generation/refinement costs are explicitly
outside the ratio (see the efficiency experiment's claim boundary,
referenced from `MASTER_REPORT.md`).

---

## Summary of what changed since v36

| Issue | Action | Artifact |
|---|---|---|
| P1a independent confirmation | **New experiment run** (frozen models on frozen independent units) | `expF_independent_confirmation/` |
| P1b target-acceptance case | Recommendation adopted (fixed-rule case selection or Supplement move); no recomputation | this document §P1b |
| P2a post-selection chronology | Narrative rule reaffirmed | `MASTER_REPORT.md` |
| P2b current8 provenance | Already audited; 5/3 split information-addition vs re-expression | `expB_T_decomposition/current8_provenance.md` |
| P2c model-family mixed | HGBR fixed as assessor; sensitivity-only framing | `expE_model_robustness/` |
| P3a J* finite protocol | No action | — |
| P3b 790× scope | Boundary enforced | `MASTER_REPORT.md` |
