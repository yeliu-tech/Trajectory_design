# exp31 — Targeted Correction Specificity Audit — Technical Report

实验日期：2026-08-17
脚本：`scripts/exp31_targeted_specificity.py`
输出目录：`experiments/history_conditioned_ftg_final/results/exp_targeted_correction_specificity/`
性质：**POST-SELECTION specificity audit** —— T（target-approach, 10D）已由 exp21 冻结的 performance–complexity 规则选定；本实验不改选择规则，只检验"被选中的 T 是否以及如何区别于其他合理工程修正"。

---

## A. One-sentence verdict

"The target-approach correction T was **not** the feature group with the strongest mechanistic correspondence to the diagnosed aliasing (design-resource R ranked highest on pair-distance↔|ΔJ*| association), but M0+T was the **only** correction that simultaneously achieved the best ranking performance (lowest top-1 regret, highest pairwise accuracy — significantly better than every alternative including the 70D full history), the largest and only significantly burden-increasing gain in high-aliasing scenarios, and a substantially lower state dimension than the full-history reference."

---

## B. Code/data audit

**复用的冻结对象（`audit_manifest.json` 锚点全部通过）：**

- matched pairs：复现 exp19 `build_within_pairs` 管线，**6,593 对**，median |ΔJ*|/σ = 0.01253，与 exp29 落盘 CSV 逐行匹配（`pairs_match_exp29_csv: true`）；T 组距离 d_T 重算与 exp29 逐字一致。
- feature groups（`group_feature_lists.json`，与代码冻结配置一致，组间无重叠变量）：
  - **G** geometry evolution，26D（history 派生：seg length / build / turn / DLS / inclination / azimuth 统计）；
  - **R** design-resource accounting，5D（`md, required_turn_arc_deg, min_md_for_turn, md_slack, tvd_ratio`）；
  - **T** target approach，10D（bearing、off-target angle、t1 相对位置/距离、`prefix_t1_min_ell`、`prefix_t1_passed`）；
  - **S** safety-corridor evolution，4D（`prefix_min_center_dist, hist_sf_mean, hist_sf_last, hist_sf_slope_per_1000m`）。
- 训练：`exp2_train_abc.train_ensemble`（HGBR 5-member，max_iter=300，seed=2024，参数未动）；评估：exp21/exp24 val 协议（`build_cell_table` + `cell_selection_table`，β=0）。
- 冻结 strata：aliasing hi/lo cut = 0.5 / 0.25；burden cut 0.5；burden 分层 = val burden 四分位（low=Q1, mid=Q2–Q3, high=Q4，n=45/91/44），与 exp24 落核一致。
- **新训模型仅 3 个**：`models/exp31/model_bg.joblib`（M0+G 51D）、`model_br.joblib`（M0+R 30D）、`model_bs.joblib`（M0+S 29D）；M0/M1/BFULL 复用 exp21 冻结模型。G/R/S 特征零非有限元素。
- **未改动**：M1 选择规则、train/val split、J* 标签、matched pairs、HGBR 超参、任何已有实验结果。

---

## C. Mechanism specificity（Experiment A，6,593 对 matched pairs）

### C.1 High-vs-low aliasing separation（median distance contrast，scenario-cluster bootstrap B=1000）

| group | dim | median d (low) | median d (high) | ratio | contrast [95% CI] |
|---|---|---|---|---|---|
| G | 26 | 2.597 | 3.056 | 1.18 | **+0.459 [+0.166, +0.736]** |
| R | 5 | 0.347 | 0.643 | 1.85 | +0.296 [+0.245, +0.359] |
| T | 10 | 0.714 | 0.948 | 1.33 | +0.234 [+0.175, +0.294] |
| S | 4 | 0.148 | 0.266 | 1.79 | +0.117 [+0.051, +0.217] |

### C.2 Distance ↔ |ΔJ*| 关联（per-scenario Spearman，跨场景 median）

| group | median ρ [95% CI] | fraction ρ>0 |
|---|---|---|
| G | +0.259 [+0.235, +0.283] | 0.92 |
| R | **+0.375 [+0.346, +0.415]** | 0.99 |
| T | +0.318 [+0.290, +0.341] | 0.97 |
| S | +0.227 [+0.196, +0.256] | 0.92 |

### C.3 Paired contrasts（T − 其他组）

| 对比 | Δ contrast [CI] | Δ median ρ [CI] |
|---|---|---|
| T−G | −0.226 [−0.498, +0.060]（n.s.） | **+0.041 [+0.016, +0.090]** |
| T−R | **−0.062 [−0.119, −0.013]** | **−0.062 [−0.081, −0.042]** |
| T−S | **+0.116 [+0.027, +0.192]** | **+0.073 [+0.045, +0.126]**（n=154，26 个场景 d_S 组内恒定致 ρ 未定义） |

**机制结论（如实，含负结果）：** T 的机制对应**不是最强**——绝对 contrast 上 G 最大（但其 CI 最宽），距离–结果关联上 **R 最强（ρ=0.375，99% 场景为正）**，且 T−R 两项指标显著为负。T 显著强于 S，ρ 上显著强于 G。**因此本论文不得声称 "T was selected because it had the strongest aliasing correlation"，也不得在 post-hoc 意义上称 T 是机制对应最强的组。**

---

## D. Pure augmentation ranking（Experiment B，同一 HGBR/同一数据/同一协议，仅特征列不同）

| arm | dim | pairwise acc [CI] | mean top-1 regret [CI] | Δregret vs M0 [CI] |
|---|---|---|---|---|
| M0 | 25 | 0.793 [0.781, 0.805] | 12.13 [9.53, 15.33] | — |
| M0+G (BG) | 51 | 0.779 [0.767, 0.792] | 15.13 [11.91, 18.56] | **−3.01 [−6.30, −0.06]（显著变差）** |
| M0+R (BR) | 30 | 0.801 [0.789, 0.813] | 10.92 [8.39, 13.89] | +1.21 [+0.60, +1.90] |
| **M0+T (BT=M1)** | **35** | **0.819 [0.808, 0.830]** | **8.53 [6.95, 10.17]** | **+3.59 [+1.58, +6.02]** |
| M0+S (BS) | 29 | 0.793 [0.780, 0.805] | 11.60 [9.21, 14.60] | +0.53 [−0.17, +1.36]（n.s.） |
| M0+G+R+T+S (BFULL) | 70 | 0.814 [0.803, 0.825] | 11.80 [9.46, 14.37] | +0.32 [−1.39, +2.25]（n.s.） |

**Paired BT vs alternatives（`ranking_paired_BT_vs_alternatives.csv`，Δregret = Regret_X − Regret_T，正值=T 更好）：**

| 对比 | Δregret [95% CI] | Δaccuracy [95% CI] |
|---|---|---|
| BT vs BG | +6.60 [+3.88, +9.85] | +0.040 [+0.034, +0.048] |
| BT vs BR | +2.38 [+0.39, +4.56] | +0.018 [+0.012, +0.024] |
| BT vs BS | +3.06 [+1.05, +5.20] | +0.027 [+0.021, +0.033] |
| BT vs BFULL | +3.27 [+1.59, +5.04] | +0.0049 [+0.0003, +0.0097] |

**全部 8 个 CI 不跨 0。** 两个值得强调的负/对照结果：
- M0+G 显著**差于** M0——把最大的 history 块直接拼进 state 反而损害 ranking（高维噪声）；
- **70D full history（BFULL）也显著差于 35D M0+T**（regret +3.27，acc +0.005）——T 以一半维度击败全历史，"更多历史信息"本身不是答案。

---

## E. High-aliasing specificity（Experiment C，burden 分层沿用 exp24 冻结四分位）

### E.1 分层 gain vs M0（mean regret gain，正值=优于 M0）

| arm | low (n=45) | mid (n=91) | high (n=44) |
|---|---|---|---|
| M0+G | −0.64 [−3.18, +1.43] | +0.16 [−2.84, +3.65] | **−11.99 [−22.08, −3.83]（显著为负）** |
| M0+R | +0.58 [−0.28, +1.46] | +1.49 [+0.49, +2.57] | +1.26 [−0.21, +2.88]（n.s.） |
| **M0+T** | +1.75 [−0.94, +5.05] | +3.39 [+0.46, +7.61] | **+5.90 [+2.45, +9.90]** |
| M0+S | +0.71 [−1.16, +3.12] | +0.61 [−0.20, +1.56] | +0.17 [−1.08, +1.35]（n.s.） |
| Full | +1.43 [−0.67, +3.64] | +1.14 [−0.88, +3.35] | −2.49 [−8.46, +3.69]（n.s.） |

### E.2 Continuous burden slopes（robust scenario-level regression，n=180）

| arm | slope [95% CI] | p |
|---|---|---|
| M0+G | **−15.04 [−28.58, −1.50]** | 0.029 |
| M0+R | +4.37 [−1.22, +9.96] | 0.125 |
| **M0+T** | **+11.91 [+2.90, +20.92]** | **0.0095** |
| M0+S | +2.47 [−2.16, +7.11] | 0.295 |
| Full | −4.28 [−14.27, +5.70] | 0.401 |

**BT 是唯一 slope 显著为正的 arm**；BG slope 显著为负。

### E.3 High-burden 层内 T vs alternatives（paired，n=44）

| 对比 | Δgain [95% CI] |
|---|---|
| BT−BG | +17.89 [+9.41, +28.17] |
| BT−BR | +4.63 [+1.10, +8.53] |
| BT−BS | +5.72 [+2.26, +9.57] |
| BT−BFULL | +8.39 [+3.75, +13.13] |

**全部 CI 不跨 0：在 aliasing 最严重的场景中，target-approach correction 的增益显著大于所有其他合理工程修正（含 70D 全历史）。**

---

## F. Performance–complexity（Experiment D）

(dim, regret) 平面上不被支配的点构成 frontier：M0 (25, 12.13) → BS (29, 11.60) → BR (30, 10.92) → **BT (35, 8.53)**。

- **M0+T 位于 performance–complexity Pareto frontier 上**，且是唯一 regret < 10 的 arm；
- BT **严格支配** BFULL（35D < 70D 且 regret 8.53 < 11.80）与 BG；
- regret 从 M0 的 12.13 降到 BT 的 8.53（−30%），维度只 +10；再继续加维度（BR/BS/BFULL 方向）不再有收益。

三维判据核对（规格第 14 节）：
1. **Problem relevance**（T 最强区分 aliased prefixes）——**不成立**（R/G 机制对应更强，见 C 节）；
2. **Decision effectiveness**（ranking regret 最低或等效最低）——**成立**（strictly 最低，CI 全不跨 0）；
3. **Representation economy**（远低于 full-history 维度达到最好性能）——**成立**（35D vs 70D，且性能更好）。

论文应把 T 的辩护写成 **2+3**（decision effectiveness + economy + severity specificity），而不是 1。

---

## G. Manuscript-ready facts（可直接写入正文的英文结果句）

1. "A post-selection specificity audit compared the frozen target-approach correction T against the three remaining engineering feature groups (geometry evolution G, design-resource accounting R, safety-corridor S) on the identical population of 6,593 M0-matched prefix pairs."
2. "Mechanistically, T was not the group most strongly associated with the diagnosed aliasing: design-resource distances showed the highest pair-level association with |ΔJ*| (median Spearman ρ = 0.375 vs 0.318 for T; paired difference −0.062, 95% CI [−0.081, −0.042])."
3. "Under the identical HGBR architecture, training data, and evaluation protocol, M0+T achieved the lowest mean top-1 regret (8.53) and the highest pairwise ranking accuracy (0.819) among all single-group augmentations, with paired improvements over every alternative significant at the 95% level (Δregret +2.38 to +6.60)."
4. "M0+T also outperformed the 70-dimensional full-history augmentation (M0+G+R+T+S) at half the state dimension (Δregret +3.27, 95% CI [+1.59, +5.04]; Δaccuracy +0.005, 95% CI [+0.0003, +0.0097]), indicating that the benefit comes from which information is added, not from adding more history per se."
5. "Augmenting M0 with the 26-dimensional geometry-evolution group significantly degraded ranking relative to M0 (Δregret −3.01, 95% CI [−6.30, −0.06])."
6. "Only M0+T showed a significantly positive burden slope (robust regression, slope +11.91 per unit aliasing burden, 95% CI [+2.90, +20.92], p = 0.0095); the geometry-augmented arm's slope was significantly negative."
7. "In the high-aliasing stratum (n = 44 scenarios), the top-1-regret gain of M0+T over M0 (+5.90, 95% CI [+2.45, +9.90]) significantly exceeded that of every alternative correction (paired Δgain +4.63 to +17.89, all CIs excluding zero)."
8. "M0+T lies on the performance–complexity Pareto frontier and strictly dominates the full-history arm (lower dimension and lower regret)."

## H. Claim boundary

**直接支持：**
- M0+T 在所有 single-group engineering corrections 中 ranking 表现最好（paired CI 全显著）；
- M0+T 的增益随 aliasing burden 增大且是唯一显著的，在 high-burden 场景显著优于所有替代修正；
- M0+T 以 35D 击败 70D 全历史，位于 Pareto frontier；
- 更多历史信息 ≠ 更好（G 显著有害，FULL 无收益）——修正的"内容选择性"成立。

**不能声称：**
- T 是与 aliasing 机制对应最强的信息（R 更强；机制对应强 ≠ ranking 收益大，这本身是一个值得写入 Discussion 的发现）；
- T is universally optimal / a mathematically minimal sufficient statistic；
- T will outperform every possible history representation（只比较了冻结的 G/R/T/S/Full 五类）；
- "T was selected because of its aliasing separation"——选择规则是 exp21 冻结的 performance–complexity 规则，本实验是 post-selection audit；
- mechanism/specificity 已经 independent external validation；
- terminal trajectory quality must improve（本实验只测 ranking，未测终端修复）。

## 附：文件清单

- 机制：`mechanism_pairs_raw.csv`、`mechanism_group_summary.csv`、`mechanism_paired_contrasts.csv`、`per_scenario_metrics.csv`
- ranking：`pure_arm_ranking_summary.csv`、`ranking_paired_BT_vs_alternatives.csv`
- burden：`burden_specificity_raw.csv`、`burden_specificity_summary.csv`、`table_burden_slopes.csv`、`table_T_vs_alternatives_high_burden.csv`、`burden_strata_edges.json`
- 复杂度：`performance_complexity.csv`
- 图：`figure_targeted_correction_600dpi.png`（3-panel：problem correspondence / performance–complexity / severity×benefit）
- 审计：`audit_manifest.json`、`group_feature_lists.json`、`global_metrics.json`、`train_timing.json`
