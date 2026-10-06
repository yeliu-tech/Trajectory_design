# T1 独立测试数据集（reviewer data）

真正未参与 M0/M1 任何开发环节的独立测试场景池。**只用于冻结后的
M0/M1 测试，不得再用于任何特征选择、阈值确定或调参。**

## 干净性依据（构造保证 + 机器核验）

- P0 provenance 审计（`../DATA_PROVENANCE_AUDIT.md`）已机检确认：所有
  历史场景池（rl_retrain 590 = agent 470+120 双射 → 全部进入 988/180
  HR 池；formal_72 / pilot_12 / diagnostic_v0 均 USED 或 UNCERTAIN），
  无干净历史场景可复用 ⇒ 本批 60 场景全部新生成。
- 新种子流 `SEED_BASE_T1 = 2500000`（与 train 910000 / agent val
  10910000 / rl_retrain 20250713 / exp6 独立 1500000 均不相交）；
  新 env_id 段：simple 流 7000–7019、coupled 流 8000–8119。
- `t1_scenarios/manifest.json` 内含与以下各池的机器核验不相交结果
  （`disjointness_ok=true`）：scenarios_agent train/val（name+seed）、
  exp6 独立 36 场景（uid+env_id）、rl_retrain scenarios（env_id）、
  988/180 HR 池（scenario_uid）。

## 生成协议（逐字沿用 exp6 独立场景协议，仅换种子流/env 段）

- 生成器：`neighboring_well_benchmark_v1/rl_retrain/
  generate_training_scenarios.py::generate_one_scenario`
  （n_seg=9、pop_size=25、n_gen=25、单靶 t1==t2）；
- 重试：attempts=1..15，`base_seed = 2500000 + attempts*131 + env_id`，
  split='val' 语义；
- 分层（各 20，共 60）：simple = neighbor_prob=0.0 流前 20；
  medium = coupled 流 difficulty=='easy' 前 20；strong = coupled 流
  difficulty ∈ {moderate, hard} 前 20（均按 env_id 升序）。

## 数据管线（与 val_hr_labels 同管线）

1. prefix 候选池：`gen_state_outcome_data.py` 主模式（与 v4_val_all
   同生成器同参数），20 组 × 4 分支 = 80 prefix/场景；
2. 标签：production adaptive 200→400（`hr_completion_oracle.
   evaluate_prefix`，nsga_mode='nested'，噪声带 0.35 近边界加倍，
   决策单元 = (scenario_uid, group_id)），标签盐 = 新盐 `hr_t1`
   （登记；与 hr_exp2/hr_exp6/hr_t2*/hr_t3 均不同）；
3. 特征：`features_v3.build_state_features_v3(step_md=10.0)` 70 列
   （`exp2_train_abc.FEATURES_C` 口径；M0 取模型 25 列、M1 取 35 列）。

## 文件

- `t1_scenarios/`：scenario 定义（scenario_XXXX.json + pareto csv +
  scenario_metadata.json + manifest.json + _gen_cache/）；
- `t1_prefix_rows.csv`：生成器原始行（44 特征 + 场景/单元字段）；
- `t1_labels_checkpoint.csv`：内容寻址 J* 标签（含 budget/seed/
  evals/五分量）；
- `t1_features_checkpoint.csv`：70D 特征（uid + params_sha256 键）；
- `t1_independent_test.csv`：**主交付表**——每行一个 prefix：
  scenario 定义键、stratum（simple/medium/strong，即 path-dependence
  分层）、group_id/branch_id（decision-unit 关系）、k_prefix
  （checkpoint depth）、44+70 特征列、hr_* 标签列、
  hr_params_sha256、hr_rebuild_ok/hr_recipe；
- `t1_manifest.json`：行数/场景数/协议/逐文件 sha256。

## 使用约束

- scenario 内 20 个 group 即 20 个 decision units（每单元 4 候选），
  统计单位 = scenario；
- `hr_rebuild_ok=0` 的行已如实保留并计数，分析时应剔除；
- 本数据集不参与训练/校准/特征选择/阈值选择。
