# exp30 — Candidate-Assessment Computational Efficiency — Technical Report

实验日期：2026-08-17
脚本：`scripts/exp30_candidate_efficiency.py`（支持 `--stage` 分段续跑）
输出目录：`experiments/history_conditioned_ftg_final/results/exp_efficiency/`
控制台日志：`logs/exp30_efficiency_console.log`

---

## A. Audit verdict

**实际复用的冻结模块（未修改任何现有代码逻辑）：**

| 用途 | 复用对象 | 来源 |
|---|---|---|
| Direct completion（production J* 协议） | `hr_completion_oracle.evaluate_prefix`，nested CRN，`budget_split(200/400)`，content seed salt=`'hr_exp2'` | exp2 标签生产线 |
| HGBR M0 / M1 冻结模型 | `models/exp21/model_m0.joblib`、`model_m1.joblib`（5-member HistGB ensemble，max_iter=300） | exp21 |
| 特征提取 | 与 exp21/exp25 完全相同的 M0(25D)/M1(35D) state 构造函数 | 冻结特征管线 |
| Experiment B K 曲线 | `exp25_val_pool_predictions.csv`（M1）、`exp28_val_pool_predictions_m0.csv`（M0）落盘预测，未重新生成 | exp25/exp28 |

**36 个 independent scenarios 的候选池未用于本实验，原因（已核实，非随意更换）：**
exp22 落盘的 independent 候选池是 **step-9 完整轨迹**；在冻结 production 协议下，对完整轨迹调用 completion oracle 会退化为**单次自评估**（不再发生 completion search），无法测量 direct completion 的真实评估成本。因此按规格启用 stratified subset。

**子集（subset_ids.csv）：**
- 18 个 validation 场景 = 6 个难度层 × 3 场景（dual: moderate / near-boundary / hard + retrain: easy / moderate / hard；dual 池难度读场景 JSON `difficulty_level`，retrain 从场景名解析）；
- 每场景 steps 1–8，每步按 label 排名取 median / q75 两个候选（确定性规则，**未使用 HGBR 预测误差挑选样本**）；
- 288 个 pick → **270 个 unique prefix**（18 个 prefix 的 median/q75 选点重合）；
- 满足规格下限（≥288 picks，难度分层等量）。

**协议冲突检查：无。** direct completion 为顺序执行（workers=1），HGBR 训练/推理单进程（HistGB 内部 OpenMP 线程）；计时基准未读取任何已缓存的 J* 作为计时结果（p_double 仅用生产档位**计数** 2793/23914，未用其计时）。

**Environment：** Windows-10-10.0.22631-SP0，Python 3.10.11，scikit-learn 1.7.2，24 核 CPU。

---

## B. Experiment A — Candidate-assessment computational efficiency

### B.1 样本与方法

- 270 个 unique prefix，同一机器、同一 Python 进程配置、同一 benchmark session。
- HGBR：warm-up 20 次（不计入）；单候选 feature extraction / inference 各重复 20 次取 median；batch（288 states 一次性预测）重复 30 次。
- Direct completion：每候选执行一次冻结 production 协议（200 档必跑，400 档由 oracle 内部规则触发）；另取 18 个 prefix 重复 3 次做 timing variability audit。
- 计时：`time.perf_counter()`；模型加载（M0 0.081s / M1 0.083s，median）、import、CSV 读取均不计入在线评估。

### B.2 Direct completion 延迟（production expected = T200 + p_double × T400，p_double = 0.1168）

| 指标 | tier 200 | tier 400 | production expected |
|---|---|---|---|
| median s/候选 | 16.52 | 31.33 | **20.41** |
| mean s/候选 | 18.67 | 35.69 | 22.83 |
| p90 s/候选 | 27.77 | 52.57 | 34.26 |
| mean evaluator calls | 218.4 | 400.4 | — |

### B.3 HGBR surrogate 延迟

| 项 | M0 | M1 |
|---|---|---|
| feature extraction median | — | 13.36 ms |
| 5-member inference median | — | 11.49 ms |
| **total online median** | 24.94 ms | **24.92 ms** |
| batch throughput | 20,907 候选/s | 20,672 候选/s |

M0 与 M1 在线成本差 <0.1%：增加 10 个 state variables 对在线评估成本的影响可忽略。

### B.4 Speedup（per-candidate: direct_production_expected / M1_total）

- **median speedup = 790×，scenario-cluster bootstrap 95% CI [735, 854]**（B=1000，seed=2024）；
- **geometric-mean speedup = 825×，95% CI [757, 899]**。

### B.5 One-time offline training cost

- 在当前 benchmark 机器上重训冻结 M1 ensemble（9,597 行 / 988 场景 / 5 成员 / 35D，参数未改）：**2.72 s**。
- 注意：exp21 表中 M1 `train_sec≈2153s` 是早期污染值（含其他开销），以本次实测 2.72s 为准。
- Amortized cost（`amortized_cost.csv`）：N=10 时 direct 204.1s vs surrogate 2.97s；N=10,000 时 direct 204,057s（≈56.7h）vs surrogate 251.9s（≈4.2min）。
- **Break-even N = 0.134**：一次完整训练的成本已低于对单个候选做一次 direct completion。

### B.6 Timing variability audit

18 prefix × 3 reps：median CV = 5.5%——direct completion 计时波动不影响结论量级。

---

## C. Experiment B — Retained-candidate budget vs ranking quality

数据源：exp25（M1）/ exp28（M0）落盘 validation 池预测，K ∈ {1,2,3,4,6,8}，random-K 100 次重采样；CI 为 scenario-cluster bootstrap（B=1000）。完整表：`budget_tradeoff_summary.csv`。

**True-best survival@K（M0 / M1 / Δ=M1−M0）：**

| K | M0 | M1 | Δ [95% CI] |
|---|---|---|---|
| 1 | 0.167 | 0.174 | +0.007 [−0.044, +0.059] |
| 2 | 0.337 | 0.315 | −0.022 [−0.089, +0.044] |
| 3 | 0.481 | 0.441 | −0.041 [−0.107, +0.022] |
| 4 | 0.556 | 0.526 | −0.030 [−0.104, +0.037] |
| 6 | 0.629 | 0.621 | −0.008 [−0.083, +0.067] |
| 8 | 0.729 | 0.754 | +0.025 [−0.042, +0.092] |

survival 差异在所有 K 上 CI 均跨 0——**不显著**。

**Retained regret（M0 / M1 / Δ=M1−M0）：**

| K | M0 | M1 | Δ [95% CI] |
|---|---|---|---|
| 1 | 7.26 | 4.25 | **−3.01 [−6.03, −0.31]** |
| 2 | 2.60 | 2.25 | −0.35 [−1.03, +0.50] |
| 3 | 1.43 | 1.02 | −0.41 [−0.90, +0.01] |
| 4 | 1.05 | 0.77 | −0.28 [−0.78, +0.11] |
| 6 | 0.59 | 0.41 | −0.19 [−0.51, +0.13] |
| 8 | 0.49 | 0.25 | −0.24 [−0.60, +0.08] |

retained regret 仅在 **K=1** 显著优于 M0，其余 K 的 CI 跨 0。

**结论（按规格不夸大）：** M1 improves retained-set quality at K=1 (retained regret −3.01, 95% CI excludes 0)；true-best survival 在任何 K 上均未显著改善；**a reduction in retained-candidate budget (same-quality smaller K) is not demonstrated.**

## D. Experiment C — NOT RUN

按规格为 OPTIONAL；需重跑 refinement pipeline，未执行。本目录不含任何 downstream evaluator-budget 结果，正文不得引用。

---

## E. Manuscript-ready facts（可直接写入论文的英文事实句）

1. "On 270 held-out trajectory prefixes from 18 validation scenarios, direct completion-to-go evaluation under the frozen production protocol cost a median of 20.4 s per candidate (expected over the adaptive 200/400 budget tiers), whereas the five-member HGBR surrogate assessed the same candidates in a median of 24.9 ms including feature extraction."
2. "This corresponds to a median per-candidate speedup of 790× (scenario-cluster bootstrap 95% CI 735–854; geometric mean 825×, 95% CI 757–899)."
3. "Batch prediction over 288 candidate states sustained roughly 20,700 candidate assessments per second for the M1 state."
4. "Extending the state from M0 (25D) to M1 (35D) changed the online assessment latency by less than 0.1% (24.94 ms vs 24.92 ms median), so the ranking gains of the target-approach correction come at negligible additional online cost."
5. "One-time offline training of the frozen five-member M1 ensemble took 2.72 s on the benchmark machine (9,597 prefix-state rows, 988 scenarios), giving a break-even point below a single direct completion evaluation."
6. "Under a fixed retained-candidate budget K, M1 lowered retained-set regret significantly at K=1 (Δ = −3.01, 95% CI [−6.03, −0.31]); true-best survival@K did not differ significantly from M0 at any tested K, and a reduction in retained-candidate budget was not demonstrated."
7. "All confidence intervals use scenario as the resampling unit (scenario-cluster bootstrap, B = 1000)."

## F. Claim boundary

**已直接验证：**
- HGBR surrogate 把 completion-to-go 评估从 ~20 s/候选 降到 ~25 ms/候选（约三个数量级）；
- M1 的 10 维修正几乎不增加在线评估成本；
- 一次性训练成本极低，摊销后 break-even < 1 个候选；
- K=1 时 M1 的 retained-set regret 显著更低。

**不能声称：**
- “HGBR 使整个轨迹优化器快 X 倍”——只测了 candidate assessment 环节；
- “M1 降低总优化成本 X%”——未测 end-to-end evaluator cost（Experiment C NOT RUN）；
- “HGBR replaces SLSQP”——两者角色不同（surrogate 排序 vs 数值修复）；
- “M1 可用更小 K 达到同等保留质量”——survival 曲线未支持；
- 速度数字不能跨机器引用（同机同 session 测量，换机需重测）。

## G. Recommended manuscript placement

- **Abstract**：可加一句效率结果（建议句 1+2 的压缩版，如 "the learned surrogate reduces per-candidate assessment cost by about three orders of magnitude"）。
- **Methodology**：AI-based candidate assessor 段末加一句在线成本构成（feature extraction + 5-member inference ≈ 25 ms/候选）。
- **Results**：新增一小节 "Candidate-assessment efficiency"（B.2–B.5 + C 段），置于 terminal boundary 之前，作为 surrogate 实用性的直接证据。
- **Figure**：`figure_efficiency_600dpi.png`（2-panel：latency + budget-survival）可作为正文候选图；`figure_budget_regret_600dpi.png` 放 Appendix。
- **Appendix**：`candidate_timing_summary.csv`、`amortized_cost.csv`、batch throughput、model-load time、timing variability audit、子集规则与 tier 构成。
