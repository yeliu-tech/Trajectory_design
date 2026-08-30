# Exp A — Full-25D M0 Checkpoint-State Aliasing Audit — Technical Report

日期：2026-08-29
脚本：`experiments/history_conditioned_ftg_final/scripts/sse_expA_full25_aliasing.py`
输出：`experiments/state_sufficiency_extension/expA_full25_aliasing/`

## A. 问题与方法

原 matched-pair 诊断（Figure 3 依据，exp19/exp29）使用 10 列匹配向量
（tgt_dist, prefix_min_sf, max_prefix_dls, remaining_md, x, y, z, inc, azi_sin, azi_cos）。
本实验把匹配空间扩展到**完整 25D M0**（`exp2_feature_groups_v3.json` B 组），
其余协议逐字冻结：同 scenario、同 k_prefix；场景内标准化（ddof=0，std=0→1）；
贪心不重复最近邻；阈值 = 组内候选对距离中位数 × {0.25, 0.5, 1.0}；
scenario-cluster bootstrap（B=1000, seed=2024）。

**有效维度说明（重要）**：M0 25 列中 k_prefix + 10 个场景常数
（max_dls_limit, deep_min_sf_threshold, min_safe_distance, target_radius_h,
target_radius_v, neighbor_count, w0, w1, w2, t1_radius_h）在全部 1,193 个
(uid,k) 匹配组内恒定，标准化后零贡献；有效匹配维 = **14** 个即时量
（x, y, z, inc, azi_sin, azi_cos, tgt_dx, tgt_dy, tgt_dz, tgt_horiz_dist,
tgt_dist, prefix_min_sf, max_prefix_dls, remaining_md——逐列核实：每列在
≥932/1193 个组内有变差）。25D 相对 10D 的新增信息是
tgt_dx/dy/dz/tgt_horiz_dist（目标误差的方向分解）与 remaining 已含的冗余表达。

**锚点校验**：同协议 10D 复算与 exp19 落盘逐字一致
（6,593 对 / median 0.01253 / P(>0.5σ)=0.12362，`full25_aliasing_anchor.json` match=true）。

## B. 关键结果（thr = 1.0× 组内中位数，主口径）

| Matching space | pairs | scenarios | median \|ΔJ*\|/σ [95% CI] | P(>0.25σ) | P(>0.5σ) [95% CI] | P(>1.0σ) |
|---|---|---|---|---|---|---|
| 10D original | 6,593 | 180 | 0.01253 [0.01032, 0.01446] | 0.1785 | 0.1236 [0.1146, 0.1326] | 0.0672 |
| **25D full M0** | **6,605** | **180** | **0.01216 [0.00999, 0.01413]** | **0.1731** | **0.1207 [0.1115, 0.1301]** | **0.0671** |

阈值敏感性（P(>0.5σ)，10D vs 25D）：
0.25× 阈值：0.0417 vs 0.0431；0.5×：0.0749 vs 0.0741；1.0×：0.1236 vs 0.1207。
三档上两个匹配空间的结果都几乎重合。

描述性：25D pair distance 与 |ΔJ*|/σ 的 Spearman ρ ≈ +0.21（弱正相关，仅描述，
不作 state-sufficiency 主证据）。

## C. 判定

**full-25D M0 aliasing 成立。** 把匹配空间从 10D 扩到完整 25D M0 后，
heavy tail 没有可辨别的衰减：median |ΔΔJ*|/σ 与三档超阈概率在两个匹配空间
间差异远小于 bootstrap CI 宽度，且阈值敏感性曲线全程平行。
论文可以继续使用"（完整）M0 checkpoint state 存在 aliasing"的强表述；
Figure 3 的 10D 匹配**不**是人为漏维度造成的假象。

## D. 对 Figure 3 的影响

- 不需要替换 Figure 3 主结论；可在正文/图注加一句：
  "the heavy tail is essentially unchanged when matching uses the full 25-dimensional
  M0 state instead of the 10-variable matching vector (P(|ΔJ*|>0.5σ): 0.121 vs 0.124)"。
- 报告有效匹配维 = 14（11 列在全部组内恒定）这一点应写入方法或附录，避免"25 个自由匹配维度"的误读。

## E. 文件

- `full25_aliasing_pairs.csv` — 25D 主配对 6,605 对（Exp C/D 的 pair 总体来源）
- `full25_aliasing_summary.csv` — 10D vs 25D 并列主表
- `full25_aliasing_sensitivity.csv` — 3 档阈值 × 2 匹配空间
- `full25_aliasing_anchor.json` — 锚点校验、描述性统计、有效维说明
- `figure_full25_aliasing.png/.pdf` — (a) ECDF (b) 阈值敏感性 (c) 距离–|ΔJ*| 散点
