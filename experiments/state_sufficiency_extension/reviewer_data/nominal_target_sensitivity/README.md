# T3 nominal target sensitivity（reviewer data）

检查 J\* 及 prefix 排序对 target 口径的敏感性：registered 运行容差
vs nominal 物理名义准则 d≤1。

## registered tolerance 来源（代码事实，详见 t3_tolerance_provenance.md）

- `scripts/a10h_v22c_multi_scenario_dataset.py:66`：
  `target_tolerance = max(1.0, env.target.radius_v / 5.0)`——
  每场景 registered 容差 = 垂直靶半径/5（下限 1.0）；
- nominal 口径：`thresholds['target_ellipsoid_norm'] = 1.0`（同文件
  :57），即 d≤1；
- 传播链：evaluator.target_tolerance → in_t1/in_t2
  （trajectory_optimizer.py:111）→ raw_metrics
  （engineering_state_evaluator.py:442）→ continuous_components 的
  target 分量（hr_completion_oracle.py:119）→ review 的
  geometry_pass / engineering_feasible。

## 协议

| regime | target_tolerance | 标签来源 |
|---|---|---|
| registered | max(1.0, radius_v/5)（场景各自值） | 引用 `val_hr_labels.csv`，**不重算** |
| nominal_d_le_1 | 1.0（唯一覆盖） | 本目录新算，盐 `hr_t3` |

nominal 重算使用同 production 策略 (B/2,B/4,B/4)、同 adaptive
200→400 加倍（噪声带 0.35、决策单元 = (scenario_uid, group_id)）、
同 nsga_mode='nested'；唯一变化 = `evaluator.target_tolerance:=1.0`。

## prefix 选择（冻结规则；不使用任何模型预测/误差）

- 与 T2 同一分层等距规则取 40 个 val 场景；
- 每场景 5 个完整 group：按 (k_prefix, group_id) 升序取分位点
  0/25/50/75/100%（去重不足按序补足）；
- 规模：40 × 5 × 4 = 800 prefixes，覆盖不同 checkpoint、J\* 水平与
  ambiguity 水平。

## 文件

- `t3_selected_prefixes.csv`：选择结果 + registered 现有标签列；
- `t3_labels_checkpoint.csv`：nominal 标签 checkpoint；
- `t3_nominal_results.csv`：**主交付长表**——prefix × regime 的
  J\* / 五分量 / budget / evals / joint_pass_ratio / 每场景
  registered_tolerance 值 / status；
- `t3_tolerance_provenance.md`：registered tolerance 来源审计；
- `t3_manifest.json`。

## 声明

- prefix 特征（含 `prefix_t1_passed` 等）与 44 列生成器特征均按
  registered 口径构建；本数据集只重算 completion-to-go 标签，
  不重建特征；
- 同一 prefix 两行用 `params_sha256` + `regime` 对齐。
