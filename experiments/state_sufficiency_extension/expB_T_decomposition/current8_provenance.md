# current8 / history2 Feature Provenance

Date: 2026-08-30. Verified against code, not inferred from names.
Source of truth: `experiments/history_conditioned_ftg_final/config/seg_state_features.py`
(batch path `_assemble_row`, line 220; incremental path `_features_from_ctx`, line 413;
ellipsoid helper `_min_ell_over`, line 210). Group definitions:
`config/structured_history_groups_v1.json` (T group, frozen).

## Coordinate convention (material for interpretation)

The 10 T variables reference **two different targets**:

- **t2** = final target: `bearing_rel_sin/cos`, `off_target_angle_deg`
- **t1** = intermediate target: `off_target1_angle_deg`, `t1_dx/dy/dz`, `t1_dist`,
  `prefix_t1_min_ell`, `prefix_t1_passed`

(Single-target scenarios set t1 = t2, so the two subgroups coincide there.)
So "current8" = 3 t2-relative + 5 t1-relative current-endpoint variables;
"history2" = 2 t1-relative prefix-history scalars.

## current8 (8D, current endpoint geometry)

| Feature | Ref | Code (seg_state_features.py) | Definition |
|---|---|---|---|
| `bearing_rel_sin` | t2 | `_assemble_row` L248–251 | sin of signed difference between current azimuth and the bearing of the t2 direction (horizontal plane) |
| `bearing_rel_cos` | t2 | `_assemble_row` L248–251 | cos of the same signed bearing difference |
| `off_target_angle_deg` | t2 | L253–260 | 3D dogleg angle (deg) between current (inc, azi) direction and the straight line endpoint→t2 |
| `off_target1_angle_deg` | t1 | L269–275 | same dogleg angle, endpoint→t1 |
| `t1_dx` | t1 | L265 | t1.x − endpoint.x |
| `t1_dy` | t1 | L266 | t1.y − endpoint.y |
| `t1_dz` | t1 | L267 | t1.z − endpoint.z |
| `t1_dist` | t1 | L268 | Euclidean norm of (t1_dx, t1_dy, t1_dz) |

All eight are functions of the **current endpoint pose + target coordinates only** — no prefix history enters. This is why Exp B interprets current8 as *re-parameterization* of information M0 partially holds in absolute coordinates (M0 already contains x/y/z/inc/azi and tgt_dx/dy/dz/dist relative to t2; the t1-relative quantities are the genuinely new geometric content in current8).

## history2 (2D, prefix-history scalars)

| Feature | Ref | Code | Definition |
|---|---|---|---|
| `prefix_t1_min_ell` | t1 | `_min_ell_over` L210; batch L181; incremental L467–478 | minimum over **all prefix points** of the normalised ellipsoid distance to t1 (radii t1_radius_h / t1_radius_v) — a path functional, not an endpoint quantity |
| `prefix_t1_passed` | t1 | L278 | binary: 1 iff `prefix_t1_min_ell <= evaluator.target_tolerance` (the evaluator's registered tolerance attribute, default 1.0 at feature-build time) |

Both are **genuinely historical**: they cannot be reconstructed from the current endpoint pose, because they depend on whether/where the prefix passed near t1 earlier. This is the Exp B "genuine path-history" component.

## Overlap check

- current8 ∩ M0: no shared columns; but `bearing_rel_*`/`off_target_angle_deg` are deterministic re-expressions of M0's (inc, azi_sin, azi_cos, tgt_dx/dy/dz) — representation overlap, not column overlap.
- history2 ∩ M0: none (no prefix t1-approach quantity exists in M0's 25 columns).
- current8 ∩ history2: none (endpoint vs path functional).

## Two code paths

- Batch feature build: `make_prefix_ctx` (L346) + `_assemble_row` (L220).
- Incremental update: `_features_from_ctx` (L413), which updates `prefix_t1_min_ell` as `min(ctx['t1_min_ell'], new-point ell)` (L467) — the running-min semantics are identical in both paths; the frozen train/val label CSVs were produced by the batch path.
