# DATA_PROVENANCE_AUDIT.md — 审稿补充数据 · 数据资产来源审计

审计日期：2026-10-06（机器事实全部由 `scripts/p0_provenance_audit.py` 只读收集，
原始输出见 `p0_audit_summary.json`；prefix/scenario 级明细见 `DATA_INVENTORY.csv`）。

判定规则（预登记，来自获批计划）：

- **USED** = 出现在 HGBR 训练 / 180 场景 state development / M1 特征选择 /
  matching·aliasing 阈值确定 / 模型或超参选择 / 36 场景分析 / 下游结果分析的任一输入；
- **CLEAN** = 可机检证明从未进入上述任一环节，且 name+seed+base geometry
  与 988/180/36 无重叠；
- **UNCERTAIN** = 无法机检证明（保守处理）。

---

## 1. 数据资产总表

| # | 数据资产 | 位置 | 规模 | 组成 / ID 范围 | 曾用于 | 判定 |
|---|---|---|---|---|---|---|
| 1 | train HR 标签池 | `experiments/history_conditioned_ftg_final/data/train_hr_labels.csv` | 10,000 行 / 988 场景（9,597 行 J\* 有效，403 行 NaN） | 470 nw_retrain（agent train 全量）+ 458 nw_dual_train（全量）+ 60 nw_formal | HGBR 训练 | **USED** |
| 2 | val HR 标签池 | `.../data/val_hr_labels.csv` | 14,400 行 / 180 场景（=180×80，无 NaN） | 120 nw_retrain（agent val 全量）+ 60 nw_dual_val（全量） | state development、M1 特征选择、验证、matching/阈值、exp26/Exp D/exp30 取样来源 | **USED** |
| 3 | 36 独立场景 | `.../data/independent_scenarios/`（manifest 在内） | 36（simple/medium/strong 各 12；env 5000-5011 / 6000-6071） | generator seed_base=1500000 | exp6 独立验证、exp15/16/17/22 下游、exp30/31、Exp F 独立排序确认 | **USED**（已作为"previous test"消耗） |
| 4 | rl_retrain 场景文件池 | `neighboring_well_benchmark_v1/rl_retrain/scenarios/` | 590 文件（env_id 2..600） | **与 agent 470+120 精确双射**（机检：两个方向差集均为空） | 经 agent 池全部进入 988/180 | **USED** |
| 5 | dual 场景池 | `experiments/segment_value_design/scenarios_dual/{train,val}/` | 458 train + 60 val | nw_dual_train_*/nw_dual_val_* | 全量进入 988/180 | **USED** |
| 6 | formal_72 | `neighboring_well_benchmark_v1/formal_72/scenarios/` | 72 | nw_formal_001..072 | 60 个在 988 train；其余 12 个出现在标签管线 demo（`results/high_resolution_prefix_labels.csv`，400 行/24 场景）、gate1 probes（4 个）、oracle demo/reconcile（nw_formal_016） | **USED** |
| 7 | pilot_12 | `neighboring_well_benchmark_v1/pilot_12/case_params/` | 12（nw_pilot_008 等） | 邻井基准早期 pilot | **未出现在本论文任何管线产物**（对 HCFTG results 全部 CSV 的 scenario_uid 机检扫描为空）；但曾被邻井基准 RL/BC 实验使用（非本论文模型） | **UNCERTAIN**（按预登记规则保守处理；且仅 12 个，量级不够） |
| 8 | diagnostic_v0 | `neighboring_well_benchmark_v1/diagnostic_v0/scenarios/` | 144（nw_001..） | 邻井基准 v0 诊断池 | 同上的缺席机检为空；历史上属邻井基准项目 | **UNCERTAIN** |
| 9 | legacy 105 对（matched pairs） | `paper_ftg_package/final_completion/results/currently_valid_matched_prefix_pairs.csv` | 105 对 / 6 场景 | general_102、general_111 + nw_dual_train_{001,003,004,007} | gate1 原始 10D aliasing 审计、exp19/exp29 复核（v15 论文 Figure S2 "legacy 105" 即此） | **USED**；注意 general_* 两个场景的**定义文件未定位**（仅有 pairs CSV），4 个 nw_dual_train 场景同时在 988 train 中 |
| 10 | exp26 收敛审计 | `.../results/exp26_label_convergence.csv` | 1,149 行 / **222 unique prefixes** / 134 场景 | budget 档 200→6000 | 标签收敛性审计（机检：134 uid 全部 ⊆ val 180） | **USED** |
| 11 | Exp D 高预算重标 | `experiments/state_sufficiency_extension/expD_highbudget_relabel/` | **862 unique prefixes** / 121 场景（budget=3200） | 来自 Exp A matched pairs（机检：121 uid 全部 ⊆ val 180） | aliasing 稳健性验证 | **USED** |
| 12 | exp30 效率子集 | `results/exp30_exp31_supplementary_20260817/exp30_efficiency/candidate_timing_raw.csv` | 270 行 / 18 场景 | val 180 子集 | 效率实验 | **USED** |
| 13 | 下游等预算/保留预算结果 | `.../results/exp15_*`、`exp16_*`、`exp17_*` 等 | scenario 级（36 场景 × arms × budgets） | — | equal-cost 下游比较 | **USED** |

## 2. 池间重叠机器校验（全部通过）

| 校验 | 结果 |
|---|---|
| 988 train 与 180 val 的 scenario_uid 交集 | **空** |
| 988 与 180 的 nw_retrain env_id 交集 | **空**（无 env_id 级泄漏） |
| agent train(470) 与 agent val(120) 的 env_id 交集 | **空** |
| agent 470+120 ↔ rl_retrain 590 文件 | **精确双射**（两方向差集均空） |
| exp26 134 uid ⊆ val 180 | 是 |
| Exp D 121 uid ⊆ val 180 | 是 |
| 36 独立场景与 470/120 的 name+seed 不重叠 | 已登记（manifest `disjointness_check.ok=true`），env 段 5000+/6000+ 与所有历史流不重叠 |
| pilot_12 / diagnostic_v0 在本论文管线产物中 | 均未出现（机检扫描为空） |

**关于"同一基础井几何是否可能同时出现在训练和测试"**：988 与 180 在 scenario name 和
nw_retrain env_id 两个键上均不重叠；nw_retrain 场景一个 env_id 对应一份几何
（name 含 geometry 类型与 seed），故不存在同一几何同时进训练与验证的情况。
36 独立场景使用独立 env 段 + 独立 seed 流，生成时已机器校验不重叠。

**关于 conformal 校准集与超参选择集是否复用**：conformal 校准
（`models/exp21/conformal_qhat_m1.json`）与 M1 特征选择均在 180 val 池上完成——
二者**确实复用同一 180 池**（这是已登记的协议事实，不是本次审计新发现的泄漏；
审稿若质疑，缓解方式 = T1 新独立测试集）。

## 3. registered target tolerance 来源（任务三审计项，只列证据）

- 公式：`registered tolerance = max(1.0, min(radius_v)/5)`，场景级属性；
- 代码来源：`neighboring_well_benchmark_v1/rl_retrain/generate_training_scenarios.py:122`；
- 登记位置：exp17/exp22 docstring（L35 附近）；36 场景批实测 ∈ [2.85, 6.93]，
  代表案例 4.3533 = radius_v/5；
- nominal d≤1：exp17/exp22 已有"终端轨迹按 d≤1 重判定"（只改判定不改搜索）；
  **d≤1 口径的 J\* completion 标签不存在** → T3 必须新算。

## 4. 对 T1（独立测试集）的结论

**不存在可直接复用的干净历史场景池：**

- rl_retrain 590 个场景已 100% 进入 988/180（双射机检）；
- dual/formal 池全量 USED；
- pilot_12（12 个）与 diagnostic_v0（144 个）虽机检缺席本论文管线，
  但按预登记规则只能判 UNCERTAIN（无法证明其历史使用与本论文开发完全无关，
  且分属更早的基准项目、生成器版本不同）；
- 36 独立场景已是"previous test"，不能二次包装成独立测试。

**决定（按获批计划的默认分支）：T1 全部新生成。**
用同一冻结生成器 `generate_training_scenarios.py::generate_one_scenario`（n_seg=9），
新 seed 流 + 新 env_id 段（与 910000/10910000/20250713/1500000 及 env 2..600、
5000-5011、6000-6071 均不重叠），生成 60 个场景（simple/medium/strong 各 20），
干净性由构造保证，并机器校验与全部历史池的 name+seed 无重叠后落盘证明。

## 5. 新增数据资产（2026-10-06 生成，全部在服务器 10.10.11.210 上执行）

| # | 数据资产 | 位置 | 规模 | 判定 |
|---|---|---|---|---|
| 14 | **T1 独立测试集** | `independent_test/` | 60 场景（simple/medium/strong 各 20）× 80 prefix = 4,800 行；标签 4,800 唯一状态（tier 200）+ 640 状态加倍 tier 400；rebuild 对账 0 失败、无标签缺失 | **CLEAN by construction**（SEED_BASE=2500000、env 段 simple 7000+/coupled 8000+；对 agent train/val name+seed、exp6 独立 36 uid+env_id、rl_retrain env_id、988/180 HR uid 全部机检不相交，`disjointness_ok=true`） |
| 15 | **T2 协议敏感性** | `protocol_sensitivity/` | 320 prefix（40 场景 × early/late 2 完整决策单元 × 4 分支，alloc easy15/moderate12/hard8/near-boundary5）× 3 协议 = 960 行；P0=引用 val 标签，P1=(B/2,B/2,0)、P2=(B/4,0,3B/4) 重算（盐 hr_t2p1/hr_t2p2） | **USED**（prefix 来自 val 180 池；仅作敏感性数据，不作独立测试） |
| 16 | **T3 nominal 靶区敏感性** | `nominal_target_sensitivity/` | 800 prefix（同 40 场景 × k 分位 5 单元 × 4 分支）× 2 口径 = 1,600 行；registered=引用 val 标签，nominal d≤1 重算（盐 hr_t3，仅覆盖 `evaluator.target_tolerance=1.0`） | **USED**（同上） |

三个数据集的 manifest（含全部文件 sha256）分别在各自目录 `t1_manifest.json` /
`t2_manifest.json` / `t3_manifest.json`；prefix 级明细已并入 `DATA_INVENTORY.csv`
（新增 7,360 行，总计 35,438 行）。

T1/T2/T3 标签协议与论文冻结口径完全一致：production adaptive 200→400、
noise band 0.35、决策单元 (scenario_uid, group_id)、`nsga_mode=nested`、
budget_split 来自 `config/adaptive_budget.json`；T2/T3 只改搜索配比 / 靶区容差，
objective、constraints、continuation protocol 其余部分不变。

## 6. 局限声明

- pilot_12 / diagnostic_v0 的 UNCERTAIN 判定是保守选择：机检只能证明它们不在
  本论文管线产物中，不能证明其生成 seed 流与 988/180 无任何构造相关性；
- general_* 场景定义文件未定位，legacy 105 对仅能作为"已消耗的分析产物"登记，
  无法重建其场景几何；
- 本审计只覆盖机检可及的落盘产物；未落盘的一次性分析（如交互式 session 中的
  临时试算）无法枚举——但所有进入论文结论的实验均有 run_manifest 支撑。
