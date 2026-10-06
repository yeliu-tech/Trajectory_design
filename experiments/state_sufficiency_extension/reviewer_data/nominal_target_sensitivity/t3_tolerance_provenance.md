# T3 registered target tolerance 来源审计（代码事实）

## 生成规则

- `scripts/a10h_v22c_multi_scenario_dataset.py:66`：
  `target_tolerance = max(1.0, env.target.radius_v / 5.0)`
  （`env_to_evaluator` → `TrajectoryEvaluator.target_tolerance`；
  每场景一个值，由垂直靶半径派生，下限 1.0）
- 场景加载链：`p32_data.load_random_scenarios_with_pareto` →
  `load_evaluator_from_original`（`scripts/a10h_v22c_multi_scenario_nn.py:146`）→ `env_to_evaluator`。

## 传播链（tolerance 影响哪些判定）

- `src/optimization/trajectory_optimizer.py:111-112`：
  `in_t1/in_t2 = ellipsoid_norm <= target_tolerance`
- `runtime/agent/engineering_state_evaluator.py:240,442`：
  `raw_metrics["target_tolerance"]`
- `experiments/history_conditioned_ftg_final/scripts/
  hr_completion_oracle.py:119`：`continuous_components` 的
  target 分量 `v = max(0, ell - tol) / tol`
- `scripts/phase32/p32m_engineering_review.py`：经 `ev["in_t1"]
  /ev["in_t2"]` → `geometry_pass` → `engineering_feasible`。

## nominal 口径

- `scripts/a10h_v22c_multi_scenario_dataset.py:57`：
  `thresholds["target_ellipsoid_norm"] = 1.0`（d≤1 名义准则）。
- 本实验 nominal 重算仅覆盖 `evaluator.target_tolerance = 1.0`，
  其余 objective / constraints / continuation protocol 不变；
  标签盐 `hr_t3`。

## 数值核验

- 本 subset 40 场景的 registered tolerance：
  min=3.0000，max=11.9044
- val 180 池：min=3.0000，max=11.9044，median=3.0000
  （本机 2026-08-29 实测；`tol == max(1.0, radius_v/5)` 对
  全部 180 场景逐位成立）
- 独立 36 场景池：min=2.8470，max=6.9318，median=5.3602
  ——即 v15 figS4 标注的 "2.85–6.93" 范围（该范围对应独立
  36 池，不是 val 180 池）
- 论文个案值 4.353252572 = radius_v(21.76626286)/5
  （v15 图 S8 `REGISTERED_TOL`，scenario_6000 同值逐位一致）。

## 声明

- prefix 特征（含 `prefix_t1_passed` 等）与 44 列生成器特征均
  按 registered 口径构建；本实验只重算 completion-to-go 标签，
  不重建特征。
