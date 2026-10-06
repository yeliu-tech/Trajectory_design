# T2 continuation-protocol sensitivity（reviewer data）

检查 J\* 及 prefix 排序是否过度依赖当前 frozen continuation search
protocol。**只改策略分配；objective / constraints / target
convention / adaptive 200→400 规则 / nested NSGA 模式全部不变。**

## 协议

| protocol | (n_sobol, n_slsqp, n_nsga) @ budget B | 标签来源 | 盐 |
|---|---|---|---|
| P0 | (B/2, B/4, B/4)（production） | 引用 `val_hr_labels.csv` 现有标签，**不重算** | hr_exp2（历史） |
| P1 | (B/2, B/2, 0)（Sobol + multi-start SLSQP 主导） | 本目录新算 | `hr_t2p1` |
| P2 | (B/4, 0, 3B/4)（NSGA-II 主导） | 本目录新算 | `hr_t2p2` |

三者同 base 200、同近边界加倍 400（决策单元 = (scenario_uid,
group_id)，leader gap ≤ 0.35 且 |active| ≥ 2，规则同 exp2）、同
`content_seed` 方案（仅盐不同）。

## prefix 选择（冻结规则；不使用任何模型预测/误差）

1. 场景：val 180 池按 difficulty 分层，最大余数法分配 40 个名额
   （easy 15 / moderate 12 / hard 8 / near-boundary 5），层内
   scenario_uid 升序等距取样；
2. 决策单元：每场景 2 个完整 group（4 分支全保留）——
   unit_early = k_prefix 最小的 group；unit_late = k_prefix 最大的
   group（并列规则见 t2_manifest.json / 脚本 docstring）；
3. 规模：40 × 2 × 4 = 320 prefixes，覆盖 early/late checkpoint 与
   不同 J\*/ambiguity 水平（选择不用任何标签或模型量）。

## 文件

- `t2_selected_prefixes.csv`：选择结果 + P0 现有标签列；
- `t2_labels_checkpoint.csv`：P1/P2 内容寻址标签 checkpoint；
- `t2_protocol_results.csv`：**主交付长表**——prefix × protocol 的
  J\* / 五分量 / budget / evals_total / joint_pass_ratio /
  n_completions / any_feasible / best_method / best_tail_params
  （最优 completion 尾段参数）/ status；
- `t2_manifest.json`：协议与逐文件 sha256。

## 声明

- P0 行为 `status='reused_production_label'`，是历史 production
  标签的直接引用；P1/P2 为同一批 prefix 的重算；
- 同一 prefix 三行用 `params_sha256` 对齐。
