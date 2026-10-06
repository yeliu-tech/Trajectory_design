# -*- coding: utf-8 -*-
"""p0_provenance_audit.py — 审稿补充数据 · 阶段 P0 数据 provenance 审计（只读）。

机械收集以下事实并落盘：
  1. 全部 scenario 池的 ID 清单（uid / env_id / seed / 文件路径）；
  2. 全部已标注数据集（HR 标签、exp26 收敛审计、Exp D 高预算重标、
     exp30 效率子集、legacy 105 对）的 prefix 级清单；
  3. 池间重叠机器校验（988 vs 180、agent==rl_retrain、exp26/expD ⊆ 180、
     36 独立场景已登记 disjointness、formal_72 未入 HR 的 12 个场景去向、
     pilot_12 / diagnostic_v0 在论文管线产物中的缺席证明）；
  4. DATA_INVENTORY.csv（14 列，prefix 级 + scenario 级）；
  5. p0_audit_summary.json（全部机器事实，供手工撰写
     DATA_PROVENANCE_AUDIT.md 引用）。

只读：不修改任何已有文件。输出到本文件所在目录的上级（reviewer_data/）。
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir,
                                     os.pardir, os.pardir))

HCFTG = os.path.join(_REPO, 'experiments', 'history_conditioned_ftg_final')
SSE = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension')

summary = {'repo': _REPO, 'pools': {}, 'labeled': {}, 'checks': {}}
inv_rows = []


def add_inv(dataset, scenario_id, prefix_id, checkpoint, data_source,
            used_training, used_state_sel, used_val, used_prev_test,
            clean_indep, jstar, protocol, target_conv, file_path):
    inv_rows.append({
        'dataset_name': dataset, 'scenario_id': scenario_id,
        'prefix_id': prefix_id, 'checkpoint': checkpoint,
        'data_source': data_source,
        'used_for_training': used_training,
        'used_for_state_selection': used_state_sel,
        'used_for_validation': used_val,
        'used_for_previous_test': used_prev_test,
        'clean_for_independent_test': clean_indep,
        'Jstar_available': jstar, 'protocol': protocol,
        'target_convention': target_conv, 'file_path': file_path})


def ls_names(d, suffix):
    p = os.path.join(_REPO, d)
    return sorted(f[:-len(suffix)] for f in os.listdir(d) if f.endswith(suffix)) \
        if os.path.isdir(os.path.join(_REPO, d)) else []


def _safe_int(x, default=-1):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def env_id_of(uid):
    m = re.match(r'nw_retrain_(\d+)_', uid)
    return int(m.group(1)) if m else None


def seed_of(uid):
    m = re.search(r'_seed(\d+)$', uid)
    return int(m.group(1)) if m else None


# ===========================================================================
# 1. scenario 池清单
# ===========================================================================
pools = {}
pools['dual_train'] = ls_names('experiments/segment_value_design/scenarios_dual/train', '.json')
pools['dual_val'] = ls_names('experiments/segment_value_design/scenarios_dual/val', '.json')
pools['formal_72'] = ls_names('neighboring_well_benchmark_v1/formal_72/scenarios', '.json')
pools['pilot_12'] = sorted(set(n.rsplit('_w', 1)[0] for n in
                               ls_names('neighboring_well_benchmark_v1/pilot_12/case_params', '.json')))
pools['diagnostic_v0'] = ls_names('neighboring_well_benchmark_v1/diagnostic_v0/scenarios', '.json')

for split in ('train', 'val'):
    meta = json.load(open(os.path.join(
        _REPO, 'experiments/agentic_trajectory_planning/scenarios_agent',
        split, 'scenario_metadata.json'), encoding='utf-8'))
    pools[f'agent_{split}'] = sorted(x['scenario_name'] for x in meta)
    summary['pools'][f'agent_{split}'] = {
        'n': len(meta),
        'env_id_range': [min(x['env_id'] for x in meta),
                         max(x['env_id'] for x in meta)]}

rl_files = ls_names('neighboring_well_benchmark_v1/rl_retrain/scenarios', '.csv')
rl_ids = sorted(int(re.match(r'scenario_(\d+)_pareto', f).group(1))
                for f in rl_files)
summary['pools']['rl_retrain_files'] = {
    'n': len(rl_ids), 'env_id_range': [rl_ids[0], rl_ids[-1]],
    'path': 'neighboring_well_benchmark_v1/rl_retrain/scenarios'}

indep_manifest = json.load(open(os.path.join(
    HCFTG, 'data', 'independent_scenarios', 'manifest.json'), encoding='utf-8'))
indep_jsons = ls_names('experiments/history_conditioned_ftg_final/data/independent_scenarios', '.json')
pools['independent_36'] = sorted(n for n in indep_jsons if n.startswith('scenario_'))

legacy_pairs_path = os.path.join(
    _REPO, 'paper_ftg_package', 'final_completion', 'results',
    'currently_valid_matched_prefix_pairs.csv')
legacy = pd.read_csv(legacy_pairs_path)
pools['legacy_general'] = sorted(legacy['scenario_uid'].unique())

for name in ('dual_train', 'dual_val', 'formal_72', 'pilot_12',
             'diagnostic_v0', 'independent_36', 'legacy_general'):
    summary['pools'][name] = {'n': len(pools[name]),
                              'sample': pools[name][:3]}

# ===========================================================================
# 2. 已标注数据集（prefix 级）
# ===========================================================================
tr = pd.read_csv(os.path.join(HCFTG, 'data', 'train_hr_labels.csv'))
va = pd.read_csv(os.path.join(HCFTG, 'data', 'val_hr_labels.csv'))
tru, vau = set(tr['scenario_uid']), set(va['scenario_uid'])

summary['labeled']['train_hr'] = {
    'rows': len(tr), 'uids': len(tru),
    'composition': dict(Counter(u.split('_seed')[0].rsplit('_', 3)[0]
                                for u in tru)),
    'path': 'experiments/history_conditioned_ftg_final/data/train_hr_labels.csv'}
summary['labeled']['val_hr'] = {
    'rows': len(va), 'uids': len(vau),
    'path': 'experiments/history_conditioned_ftg_final/data/val_hr_labels.csv'}

for uid, g in tr.groupby('scenario_uid'):
    for _, r in g.iterrows():
        add_inv('train_hr_labels_988', uid, r['hr_params_sha256'],
                _safe_int(r['k_prefix']), 'hr_completion_oracle production',
                True, False, False, False, False, True,
                f"production_adaptive_200_400 (hr_budget={_safe_int(r['hr_budget'])})",
                'registered_per_scenario',
                'experiments/history_conditioned_ftg_final/data/train_hr_labels.csv')

for uid, g in va.groupby('scenario_uid'):
    for _, r in g.iterrows():
        add_inv('val_hr_labels_180', uid, r['hr_params_sha256'],
                _safe_int(r['k_prefix']), 'hr_completion_oracle production',
                False, True, True, False, False, True,
                f"production_adaptive_200_400 (hr_budget={_safe_int(r['hr_budget'])})",
                'registered_per_scenario',
                'experiments/history_conditioned_ftg_final/data/val_hr_labels.csv')

# exp26 收敛审计
e26 = pd.read_csv(os.path.join(HCFTG, 'results', 'exp26_label_convergence.csv'))
e26_uids = set(e26['scenario_uid'])
summary['labeled']['exp26_convergence'] = {
    'rows': len(e26), 'unique_prefixes': int(e26['params_sha256'].nunique()),
    'uids': len(e26_uids),
    'budgets': sorted(int(b) for b in e26['budget'].unique()),
    'path': 'experiments/history_conditioned_ftg_final/results/exp26_label_convergence.csv'}
for _, r in e26.iterrows():
    add_inv('exp26_convergence_audit', r['scenario_uid'], r['params_sha256'],
            _safe_int(r['k_prefix']), 'hr_completion_oracle fixed tiers',
            False, True, True, False, False, True,
            f"fixed_budget_{int(r['budget'])}", 'registered_per_scenario',
            'experiments/history_conditioned_ftg_final/results/exp26_label_convergence.csv')

# Exp D 高预算重标
expd_path = os.path.join(SSE, 'expD_highbudget_relabel', 'unique_prefixes.csv')
expd = pd.read_csv(expd_path)
expd_uids = set(expd['scenario_uid'])
summary['labeled']['expD_highbudget'] = {
    'unique_prefixes': len(expd), 'uids': len(expd_uids),
    'path': 'experiments/state_sufficiency_extension/expD_highbudget_relabel/unique_prefixes.csv'}
for _, r in expd.iterrows():
    add_inv('expD_highbudget_relabel_3200', r['scenario_uid'],
            r['params_sha256'], _safe_int(r.get('k_prefix', -1)),
            'hr_completion_oracle fixed 3200',
            False, True, True, False, False, True,
            'fixed_budget_3200', 'registered_per_scenario',
            'experiments/state_sufficiency_extension/expD_highbudget_relabel/unique_prefixes.csv')

# exp30 效率子集（可选来源）
e30_dir = os.path.join(_REPO, 'results', 'exp30_exp31_supplementary_20260817',
                       'exp30_efficiency')
e30_csv = None
for cand in ('candidate_timing_raw.csv',):
    p = os.path.join(e30_dir, cand)
    if os.path.exists(p):
        e30_csv = p
if e30_csv:
    e30 = pd.read_csv(e30_csv)
    summary['labeled']['exp30_efficiency'] = {
        'rows': len(e30), 'uids': int(e30['scenario_uid'].nunique()),
        'path': os.path.relpath(e30_csv, _REPO)}
    for _, r in e30.iterrows():
        add_inv('exp30_efficiency_subset', r['scenario_uid'],
                r['params_sha256'], _safe_int(r['step']), 'exp30 direct timing',
                False, True, True, False, False, True,
                f"production_tiers_200_400 (evals_200={_safe_int(r.get('direct_200_evals'))},"
                f" evals_400={_safe_int(r.get('direct_400_evals'))})",
                'registered_per_scenario',
                os.path.relpath(e30_csv, _REPO))

# legacy 105 对（pair 级，记 prefix 未知 → 记 pair 首前缀 id）
for _, r in legacy.iterrows():
    add_inv('legacy_105_matched_pairs', r['scenario_uid'],
            f"pair_a{int(r['a_prefix_id'])}", _safe_int(r['k_prefix']),
            'final_completion gate1 matched-pair audit',
            False, False, False, True, False, True,
            'legacy_gate1_production', 'legacy_thresholds',
            'paper_ftg_package/final_completion/results/currently_valid_matched_prefix_pairs.csv')

# 36 独立场景（scenario 级，下游结果在 exp15/16/17 等）
for n in pools['independent_36']:
    add_inv('independent_36', n, '', -1,
            'generate_training_scenarios.py seed_base=1500000',
            False, False, False, True, False, True,
            'downstream_exp6_15_16_17_22_30', 'registered_per_scenario',
            'experiments/history_conditioned_ftg_final/data/independent_scenarios/')

# 其余 scenario 池（scenario 级）
pool_flag = {
    'dual_train': (True, False, False, False),
    'dual_val': (False, True, True, False),
    'formal_72': (True, False, False, False),
    'agent_train': (True, False, False, False),
    'agent_val': (False, True, True, False),
    'pilot_12': (False, False, False, False),
    'diagnostic_v0': (False, False, False, False),
}
pool_path = {
    'dual_train': 'experiments/segment_value_design/scenarios_dual/train/',
    'dual_val': 'experiments/segment_value_design/scenarios_dual/val/',
    'formal_72': 'neighboring_well_benchmark_v1/formal_72/scenarios/',
    'agent_train': 'experiments/agentic_trajectory_planning/scenarios_agent/train/',
    'agent_val': 'experiments/agentic_trajectory_planning/scenarios_agent/val/',
    'pilot_12': 'neighboring_well_benchmark_v1/pilot_12/case_params/',
    'diagnostic_v0': 'neighboring_well_benchmark_v1/diagnostic_v0/scenarios/',
}
for pname, (ut, us, uv, up) in pool_flag.items():
    for n in pools[pname]:
        add_inv(f'pool_{pname}', n, '', -1, 'scenario pool (unlabeled here)',
                ut, us, uv, up, False, False, 'n/a', 'n/a', pool_path[pname])

# ===========================================================================
# 3. 池间重叠机器校验
# ===========================================================================
checks = {}
checks['hr_name_overlap'] = sorted(tru & vau)
tr_ids = {env_id_of(u) for u in tru if env_id_of(u) is not None}
va_ids = {env_id_of(u) for u in vau if env_id_of(u) is not None}
checks['retrain_env_id_overlap_988_180'] = sorted(tr_ids & va_ids)
checks['exp26_uids_not_in_val180'] = sorted(e26_uids - vau)
checks['expD_uids_not_in_val180'] = sorted(expd_uids - vau)

agent_ids = set()
for split in ('train', 'val'):
    meta = json.load(open(os.path.join(
        _REPO, 'experiments/agentic_trajectory_planning/scenarios_agent',
        split, 'scenario_metadata.json'), encoding='utf-8'))
    agent_ids |= {x['env_id'] for x in meta}
checks['agent_env_ids_minus_rlfiles'] = sorted(agent_ids - set(rl_ids))
checks['rlfiles_minus_agent_env_ids'] = sorted(set(rl_ids) - agent_ids)
checks['agent_train_env_overlap_val'] = None  # 下方填

meta_tr = json.load(open(os.path.join(
    _REPO, 'experiments/agentic_trajectory_planning/scenarios_agent/train',
    'scenario_metadata.json'), encoding='utf-8'))
meta_va = json.load(open(os.path.join(
    _REPO, 'experiments/agentic_trajectory_planning/scenarios_agent/val',
    'scenario_metadata.json'), encoding='utf-8'))
checks['agent_train_env_overlap_val'] = sorted(
    {x['env_id'] for x in meta_tr} & {x['env_id'] for x in meta_va})

checks['independent_36_disjointness_registered'] = indep_manifest.get(
    'disjointness_check', {}).get('ok')
checks['independent_36_env_ranges'] = indep_manifest.get('env_id_ranges')

formal_not_hr = sorted(set(pools['formal_72']) - tru)
checks['formal_72_not_in_hr_train'] = formal_not_hr
# 这 12 个在下游实验中的使用证据
usage_evidence = {}
for fn in ('exp15_raw_cells.csv', 'exp16_replay_results.csv'):
    p = os.path.join(HCFTG, 'results', fn)
    if os.path.exists(p):
        d = pd.read_csv(p, usecols=lambda c: c == 'scenario_uid')
        if 'scenario_uid' in d.columns:
            usage_evidence[fn] = sorted(set(formal_not_hr) & set(d['scenario_uid']))
checks['formal_72_12_usage_in_downstream'] = usage_evidence

# pilot_12 / diagnostic_v0 在论文管线产物中的缺席扫描
pipeline_uids = tru | vau | e26_uids | expd_uids
for fn in os.listdir(os.path.join(HCFTG, 'results')):
    if fn.endswith('.csv'):
        try:
            d = pd.read_csv(os.path.join(HCFTG, 'results', fn),
                            usecols=lambda c: c == 'scenario_uid')
            if 'scenario_uid' in d.columns:
                pipeline_uids |= set(d['scenario_uid'].dropna())
        except Exception:
            pass
checks['pilot_12_in_pipeline'] = sorted(set(pools['pilot_12']) & pipeline_uids)
checks['diagnostic_v0_in_pipeline'] = sorted(set(pools['diagnostic_v0']) & pipeline_uids)
checks['legacy_general_in_hr_pools'] = sorted(set(pools['legacy_general']) & (tru | vau))

summary['checks'] = checks

# ===========================================================================
# 4. 落盘
# ===========================================================================
inv = pd.DataFrame(inv_rows)
inv_path = os.path.join(OUT_DIR, 'DATA_INVENTORY.csv')
inv.to_csv(inv_path, index=False, encoding='utf-8-sig')
sum_path = os.path.join(OUT_DIR, 'p0_audit_summary.json')
with open(sum_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f'[done] DATA_INVENTORY.csv rows={len(inv)} -> {inv_path}')
print(f'[done] p0_audit_summary.json -> {sum_path}')
print('[checks]', json.dumps({k: (v if not isinstance(v, list) or len(v) < 6
                                  else f'<{len(v)} items>')
                              for k, v in checks.items()},
                             ensure_ascii=False, indent=1, default=str))
