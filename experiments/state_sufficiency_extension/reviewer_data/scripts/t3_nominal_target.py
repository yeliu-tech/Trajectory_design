# -*- coding: utf-8 -*-
"""
t3_nominal_target.py — 审稿补充数据 T3：nominal target sensitivity
（同一批 val prefix 在 registered 与 d≤1 两套 target 口径下的 J*）。

=====================================================================
登记（2026-08-29，运行前定稿；此后不改抽样规则、不改协议、不改阈值
覆盖方式、不改输出 schema）
=====================================================================

【registered tolerance 来源审计（代码事实，非解释）】
  - scripts/a10h_v22c_multi_scenario_dataset.py:66
        target_tolerance = max(1.0, env.target.radius_v / 5.0)
    即每场景 registered 运行容差 = 垂直靶半径 / 5（下限 1.0），
    经 env_to_evaluator → TrajectoryEvaluator.target_tolerance。
  - 传播链：evaluator.target_tolerance
      → src/optimization/trajectory_optimizer.py:111  in_t1/in_t2
      → runtime/agent/engineering_state_evaluator.py:240,442
        raw_metrics['target_tolerance']
      → hr_completion_oracle.continuous_components :119  target 分量
      → scripts/phase32/p32m_engineering_review.py  in_t1/in_t2
        → geometry_pass / engineering_feasible。
  - nominal 口径：thresholds['target_ellipsoid_norm'] = 1.0
    （scripts/a10h_v22c_multi_scenario_dataset.py:57）——即 d≤1 的
    物理名义准则；本实验将 evaluator.target_tolerance 覆盖为 1.0，
    其余 objective / constraints / continuation protocol 全部不变。
  - 论文中 4.3533 = 个案场景 radius_v/5（例：scenario_6000
    radius_v=21.76626286 → 4.353252572，与 v15 图 S8
    REGISTERED_TOL 逐位一致）；2.85–6.93 为独立 36 场景池的范围
    （v15 figS4 标注；val 180 池实测为 3.00–11.90）。

【协议】
  registered：引用 val_hr_labels.csv 现有 production 标签（不重算）。
  nominal   ：同 production 策略 (B/2,B/4,B/4)、同 adaptive 200→400
              加倍规则（噪声带 0.35、决策单元=(scenario_uid,
              group_id)）、同 nsga_mode='nested'；唯一变化 =
              evaluator.target_tolerance := 1.0；salt 'hr_t3'。
  注意（报告声明）：prefix 特征（如 prefix_t1_passed）与 44 列生成
  器特征均按 registered 口径构建，本实验只重算 completion-to-go
  标签，不重建特征。

【prefix 选择（冻结规则；不使用任何模型预测/误差）】
  1. 场景：与 T2 同一分层等距规则取 40 个 val 场景。
  2. 决策单元：每场景 5 个完整 group——按 (k_prefix, group_id) 升序
     取分位点 0/25/50/75/100%（最近整数位置，去重后不足 5 个时按序
     补足），覆盖 early/middle/late checkpoint 与不同 J* 水平。
  3. 规模：40 × 5 × 4 = 800 prefixes（在 500–1000 内）。

【输出】nominal_target_sensitivity/
  t3_selected_prefixes.csv
  t3_labels_checkpoint.csv        nominal 标签 checkpoint（断点续跑）
  t3_nominal_results.csv          长表：prefix × {registered, nominal}
  t3_tolerance_provenance.md      registered tolerance 来源审计
  t3_manifest.json

用法：
  python t3_nominal_target.py --stage all --workers 64
  分阶段：--stage select|labels|assemble
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RD_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_SSE_DIR = os.path.abspath(os.path.join(_RD_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_SSE_DIR, os.pardir, os.pardir))
_HCFTG_SCRIPTS = os.path.join(
    _REPO_ROOT, 'experiments', 'history_conditioned_ftg_final', 'scripts')
if _HCFTG_SCRIPTS not in sys.path:
    sys.path.insert(0, _HCFTG_SCRIPTS)

import exp2_relabel_train_val as e2r  # noqa: E402  # 模块级 pin repo root

# ---------------- T3 登记数值（冻结） ----------------
N_SCENARIOS = 40
UNITS_PER_SCENARIO = 5
NOMINAL_TOLERANCE = 1.0
BASE_BUDGET = 200
DOUBLE_BUDGET = 400
NOISE_BAND = 0.35
N_SEG = 9
SALT = 'hr_t3'

VAL_CSV = os.path.join(_REPO_ROOT, 'experiments',
                       'history_conditioned_ftg_final', 'data',
                       'val_hr_labels.csv')
OUT_DIR = os.environ.get('T3_OUT_DIR',
                         os.path.join(_RD_DIR, 'nominal_target_sensitivity'))
SELECT_CSV = os.path.join(OUT_DIR, 't3_selected_prefixes.csv')
LABEL_CKPT = os.path.join(OUT_DIR, 't3_labels_checkpoint.csv')
OUT_CSV = os.path.join(OUT_DIR, 't3_nominal_results.csv')
PROV_MD = os.path.join(OUT_DIR, 't3_tolerance_provenance.md')
OUT_MANIFEST = os.path.join(OUT_DIR, 't3_manifest.json')

LABEL_FIELDS = list(e2r.LABEL_FIELDS)


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _append_rows(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
        f.flush()


# =============================================================================
# 阶段 1：prefix 选择（冻结规则；与 T2 同场景规则）
# =============================================================================

def select_prefixes():
    df = pd.read_csv(VAL_CSV)
    df = df[np.isfinite(df['hr_J_star_cont'])].copy()
    scen = df.groupby('scenario_uid').agg(
        difficulty=('difficulty', 'first')).reset_index()
    strata = {d: g['scenario_uid'].sort_values().tolist()
              for d, g in scen.groupby('difficulty')}
    total = sum(len(v) for v in strata.values())
    raw = {d: len(v) / total * N_SCENARIOS for d, v in strata.items()}
    alloc = {d: int(np.floor(x)) for d, x in raw.items()}
    rem = N_SCENARIOS - sum(alloc.values())
    for d in sorted(raw, key=lambda k: raw[k] - np.floor(raw[k]),
                    reverse=True)[:rem]:
        alloc[d] += 1
    picked = []
    for d, uids in sorted(strata.items()):
        n = alloc[d]
        idx = np.linspace(0, len(uids) - 1, n + 2, dtype=int)[1:-1]
        picked += [uids[i] for i in sorted(set(idx))]
    picked = sorted(set(picked))
    log(f'[select] 场景 {len(picked)} 个（alloc={alloc}）')

    rows = []
    for uid in picked:
        sub = df[df['scenario_uid'] == uid]
        g = sub.groupby('group_id').agg(
            k=('k_prefix', 'first'), size=('k_prefix', 'size'))
        g = g[g['size'] == 4].sort_values(['k'])
        g = g.reset_index().sort_values(['k', 'group_id'])
        if len(g) < UNITS_PER_SCENARIO:
            log(f'[select][WARN] {uid} 完整单元不足：{len(g)}')
        pos = np.linspace(0, len(g) - 1,
                          min(UNITS_PER_SCENARIO, len(g)))
        sel_idx = sorted(set(int(round(x)) for x in pos))
        # 去重后不足则按序补足
        for i in range(len(g)):
            if len(sel_idx) >= min(UNITS_PER_SCENARIO, len(g)):
                break
            if i not in sel_idx:
                sel_idx.append(i)
        sel_idx = sorted(sel_idx)
        for rank, gi in enumerate(sel_idx):
            gid = int(g.iloc[gi]['group_id'])
            unit = sub[sub['group_id'] == gid]
            for r in unit.itertuples(index=False):
                rows.append({
                    'scenario_uid': uid,
                    'difficulty': str(r.difficulty),
                    'group_id': gid,
                    'unit_rank': rank,
                    'branch_id': int(r.branch_id),
                    'k_prefix': int(r.k_prefix),
                    'n_seg_used': int(r.n_seg_used),
                    'hr_params_sha256': str(r.hr_params_sha256),
                    'registered_J_star_cont': float(r.hr_J_star_cont),
                    'registered_budget': float(r.hr_budget),
                    'registered_evals_total': float(r.hr_evals_total),
                    'registered_joint_pass_ratio':
                        float(r.hr_joint_pass_ratio),
                    'registered_n_completions': float(r.hr_n_completions),
                    'registered_any_feasible': float(r.hr_any_feasible),
                })
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(SELECT_CSV, index=False)
    log(f'[select] {len(out)} prefixes / {out.scenario_uid.nunique()} 场景 '
        f'-> {SELECT_CSV}')
    return out


# =============================================================================
# 阶段 2：nominal d≤1 标签（唯一变化 = target_tolerance 覆盖）
# =============================================================================

_W = {}


def _worker_init():
    e2r._pin_repo_root()
    if _HCFTG_SCRIPTS not in sys.path:
        sys.path.insert(0, _HCFTG_SCRIPTS)
    import hr_completion_oracle as orc  # noqa: PLC0415
    exp1 = orc._import_module(orc.EXP1_SCRIPT, 'exp1_vp')
    from scripts.phase32.p32m_engineering_review import review_trajectory
    pools = e2r.load_scenario_pools()
    registered_tol = {}
    for uid, sc in pools.items():
        ev = sc.get('evaluator')
        registered_tol[uid] = float(getattr(ev, 'target_tolerance', 1.0))
        ev.target_tolerance = NOMINAL_TOLERANCE   # 唯一覆盖
    _W.update({'orc': orc, 'exp1': exp1, 'review_fn': review_trajectory,
               'pools': pools, 'registered_tol': registered_tol})


def _eval_one(task):
    """task = (uid, psha, budget, params_tuple)。exp2._eval_one 同款，
    nominal 口径（worker 已覆盖 tolerance），盐 'hr_t3'。"""
    uid, psha, budget, params_t = task
    orc, exp1 = _W['orc'], _W['exp1']
    sc = _W['pools'][uid]
    prefix = np.asarray(params_t, dtype=np.float64)
    seed, _ = orc.content_seed(uid, prefix, salt=SALT)
    t0 = time.time()

    if len(prefix) // 3 >= exp1.N_SEG:
        counter = {'n': 0}
        ev, st = exp1._full_eval(sc, prefix, counter)
        comps = [(st, ev, 'self', np.zeros(0))] if st is not None else []
        recs = orc._completion_records(exp1, sc, prefix, comps,
                                       _W['review_fn'])
        agg = orc._aggregate(recs)
        agg.update({'evals_total': counter['n'], 'n_errors': 0})
    else:
        recs, agg = orc.evaluate_prefix(
            exp1, sc, prefix, seed, *orc.budget_split(budget),
            _W['review_fn'], nsga_mode='nested')

    row = {
        'params_sha256': psha, 'budget': int(budget), 'seed': int(seed),
        'scenario_uid': uid,
        'J_star_cont': agg['J_star_cont'],
        'J_star_old': agg['J_star_old'],
        'R_k_eng': agg['R_k_eng'],
        'joint_pass_ratio': agg['joint_pass_ratio'],
        'n_completions': agg['n_completions'],
        'n_feasible': agg['n_feasible'],
        'any_feasible': agg['any_feasible'],
        'evals_total': agg['evals_total'],
        'n_errors': agg['n_errors'],
        'elapsed_sec': round(time.time() - t0, 2),
    }
    for c in ('cont_target', 'cont_safety', 'cont_dls',
              'cont_constructability', 'cont_mechanical'):
        row[c] = float('nan')
    if recs:
        b = recs[int(np.argmin([r['cont_total'] for r in recs]))]
        for c in ('cont_target', 'cont_safety', 'cont_dls',
                  'cont_constructability', 'cont_mechanical'):
            row[c] = b[c]
    return row


def _done_label_keys():
    keys = set()
    if os.path.exists(LABEL_CKPT):
        with open(LABEL_CKPT, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                keys.add((r['params_sha256'], int(r['budget'])))
    return keys


def _load_labels():
    if not os.path.exists(LABEL_CKPT):
        return pd.DataFrame(columns=LABEL_FIELDS)
    return pd.read_csv(LABEL_CKPT)


def _run_tasks(tasks, ex, flush_every=50, flush_secs=120):
    n_new = 0
    if not tasks:
        return n_new
    buf, t_last = [], time.time()
    futs = {ex.submit(_eval_one, t): t for t in tasks}
    for fut in as_completed(futs):
        buf.append(fut.result())
        n_new += 1
        if len(buf) >= flush_every or (time.time() - t_last) >= flush_secs:
            _append_rows(LABEL_CKPT, buf, LABEL_FIELDS)
            buf.clear()
            t_last = time.time()
        if n_new % 200 == 0 or n_new == len(tasks):
            log(f'  [eval] {n_new}/{len(tasks)}')
    if buf:
        _append_rows(LABEL_CKPT, buf, LABEL_FIELDS)
    return n_new


def stage_labels(workers):
    sel = pd.read_csv(SELECT_CSV)
    pools = e2r.load_scenario_pools()
    states, row_meta = {}, {}
    n_fail = 0
    for r in sel.itertuples(index=False):
        uid = str(r.scenario_uid)
        sc = pools.get(uid)
        rowd = {'scenario_uid': uid, 'group_id': int(r.group_id),
                'branch_id': int(r.branch_id), 'k_prefix': int(r.k_prefix),
                'n_seg_used': int(r.n_seg_used)}
        params, recipe = e2r.rebuild_params(rowd, sc)
        psha = e2r._params_sha(params) if params is not None else ''
        if psha != str(r.hr_params_sha256):
            n_fail += 1
            log(f'[labels][WARN] sha 不一致 {uid} g{r.group_id} '
                f'b{r.branch_id} recipe={recipe}（剔除）')
            continue
        if psha not in states:
            states[psha] = (uid, tuple(float(x) for x in params))
        row_meta.setdefault(psha, set()).add((uid, int(r.group_id)))
    log(f'[labels] 唯一状态 {len(states)} 个（sha 对账失败 {n_fail}）')

    done = _done_label_keys()

    def submit_tier(ex, budget, only=None):
        tasks = []
        for psha, (uid, pt) in states.items():
            if only is not None and psha not in only:
                continue
            if (psha, int(budget)) not in done:
                tasks.append((uid, psha, int(budget), pt))
        log(f'[labels] 档 {budget}: 待跑 {len(tasks)}')
        _run_tasks(tasks, ex)

    with ProcessPoolExecutor(max_workers=int(workers),
                             initializer=_worker_init) as ex:
        submit_tier(ex, BASE_BUDGET)
        labels = _load_labels()
        lab200 = labels[labels['budget'] == BASE_BUDGET].set_index(
            'params_sha256')['J_star_cont']
        unit_best = {}
        for psha, units in row_meta.items():
            if psha not in lab200.index:
                continue
            j = float(lab200[psha])
            for u in units:
                unit_best[u] = min(unit_best.get(u, float('inf')), j)
        active = set()
        for psha, units in row_meta.items():
            if psha not in lab200.index:
                continue
            j = float(lab200[psha])
            for u in units:
                if j - unit_best[u] <= NOISE_BAND + 1e-12:
                    active.add(psha)
        unit_count = {}
        for psha, units in row_meta.items():
            if psha in active:
                for u in units:
                    unit_count[u] = unit_count.get(u, 0) + 1
        double = {p for p in active
                  for u in row_meta[p] if unit_count.get(u, 0) >= 2}
        double = {p for p in double
                  if len(states[p][1]) // 3 < N_SEG}
        log(f'[labels] 加倍状态 {len(double)} 个')
        submit_tier(ex, DOUBLE_BUDGET, only=double)
    log('[labels] 完成')
    return 0


# =============================================================================
# 阶段 3：装配 + tolerance provenance
# =============================================================================

def stage_assemble():
    sel = pd.read_csv(SELECT_CSV)
    labels = _load_labels()

    # ---- 每场景 registered tolerance（代码事实落盘）----
    pools = e2r.load_scenario_pools()
    tol_map = {uid: float(getattr(sc.get('evaluator'),
                                  'target_tolerance', 1.0))
               for uid, sc in pools.items()}

    records = []
    for r in sel.itertuples(index=False):
        psha = str(r.hr_params_sha256)
        base = {'scenario_uid': r.scenario_uid,
                'difficulty': r.difficulty,
                'group_id': int(r.group_id),
                'unit_rank': int(r.unit_rank),
                'branch_id': int(r.branch_id),
                'k_prefix': int(r.k_prefix),
                'params_sha256': psha,
                'registered_tolerance': tol_map.get(
                    str(r.scenario_uid), float('nan'))}
        records.append({**base, 'regime': 'registered',
                        'target_tolerance': base['registered_tolerance'],
                        'J_star_cont': r.registered_J_star_cont,
                        'budget': r.registered_budget,
                        'evals_total': r.registered_evals_total,
                        'joint_pass_ratio': r.registered_joint_pass_ratio,
                        'n_completions': r.registered_n_completions,
                        'any_feasible': r.registered_any_feasible,
                        'status': 'reused_production_label'})
        sub = labels[labels['params_sha256'] == psha]
        if sub.empty:
            records.append({**base, 'regime': 'nominal_d_le_1',
                            'target_tolerance': NOMINAL_TOLERANCE,
                            'status': 'missing'})
            continue
        b = sub.loc[sub['budget'].idxmax()]
        records.append({**base, 'regime': 'nominal_d_le_1',
                        'target_tolerance': NOMINAL_TOLERANCE,
                        'J_star_cont': float(b['J_star_cont']),
                        'cont_target': float(b['cont_target']),
                        'cont_safety': float(b['cont_safety']),
                        'cont_dls': float(b['cont_dls']),
                        'cont_constructability':
                            float(b['cont_constructability']),
                        'cont_mechanical': float(b['cont_mechanical']),
                        'budget': float(b['budget']),
                        'seed': float(b['seed']),
                        'evals_total': float(b['evals_total']),
                        'joint_pass_ratio': float(b['joint_pass_ratio']),
                        'n_completions': float(b['n_completions']),
                        'n_feasible': float(b['n_feasible']),
                        'any_feasible': float(b['any_feasible']),
                        'n_errors': float(b['n_errors']),
                        'status': 'ok'})
    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    log(f'[out] {len(out)} 行 -> {OUT_CSV}')

    # ---- provenance 文档（只写代码事实与分布，不做主观解释）----
    sel_tols = sorted({tol_map.get(str(u), float('nan'))
                       for u in sel.scenario_uid.unique()})
    lines = [
        '# T3 registered target tolerance 来源审计（代码事实）',
        '',
        '## 生成规则',
        '',
        '- `scripts/a10h_v22c_multi_scenario_dataset.py:66`：',
        '  `target_tolerance = max(1.0, env.target.radius_v / 5.0)`',
        '  （`env_to_evaluator` → `TrajectoryEvaluator.target_tolerance`；',
        '  每场景一个值，由垂直靶半径派生，下限 1.0）',
        '- 场景加载链：`p32_data.load_random_scenarios_with_pareto` →',
        '  `load_evaluator_from_original`（`scripts/a10h_v22c_multi_'
        'scenario_nn.py:146`）→ `env_to_evaluator`。',
        '',
        '## 传播链（tolerance 影响哪些判定）',
        '',
        '- `src/optimization/trajectory_optimizer.py:111-112`：',
        '  `in_t1/in_t2 = ellipsoid_norm <= target_tolerance`',
        '- `runtime/agent/engineering_state_evaluator.py:240,442`：',
        '  `raw_metrics["target_tolerance"]`',
        '- `experiments/history_conditioned_ftg_final/scripts/',
        '  hr_completion_oracle.py:119`：`continuous_components` 的',
        '  target 分量 `v = max(0, ell - tol) / tol`',
        '- `scripts/phase32/p32m_engineering_review.py`：经 `ev["in_t1"]',
        '  /ev["in_t2"]` → `geometry_pass` → `engineering_feasible`。',
        '',
        '## nominal 口径',
        '',
        '- `scripts/a10h_v22c_multi_scenario_dataset.py:57`：',
        '  `thresholds["target_ellipsoid_norm"] = 1.0`（d≤1 名义准则）。',
        '- 本实验 nominal 重算仅覆盖 `evaluator.target_tolerance = 1.0`，',
        '  其余 objective / constraints / continuation protocol 不变；',
        '  标签盐 `hr_t3`。',
        '',
        '## 数值核验',
        '',
        f'- 本 subset 40 场景的 registered tolerance：',
        f'  min={min(sel_tols):.4f}，max={max(sel_tols):.4f}',
        '- val 180 池：min=3.0000，max=11.9044，median=3.0000',
        '  （本机 2026-08-29 实测；`tol == max(1.0, radius_v/5)` 对',
        '  全部 180 场景逐位成立）',
        '- 独立 36 场景池：min=2.8470，max=6.9318，median=5.3602',
        '  ——即 v15 figS4 标注的 "2.85–6.93" 范围（该范围对应独立',
        '  36 池，不是 val 180 池）',
        '- 论文个案值 4.353252572 = radius_v(21.76626286)/5',
        '  （v15 图 S8 `REGISTERED_TOL`，scenario_6000 同值逐位一致）。',
        '',
        '## 声明',
        '',
        '- prefix 特征（含 `prefix_t1_passed` 等）与 44 列生成器特征均',
        '  按 registered 口径构建；本实验只重算 completion-to-go 标签，',
        '  不重建特征。',
    ]
    with open(PROV_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    log(f'[out] provenance -> {PROV_MD}')

    manifest = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'T3 nominal target sensitivity (d<=1)',
        'n_scenarios': int(sel.scenario_uid.nunique()),
        'n_prefixes': int(len(sel)),
        'selection_rule': '40 scenarios stratified-systematic by '
                          'difficulty (same rule as T2); per scenario 5 '
                          'complete units at k-prefix quantiles '
                          '0/25/50/75/100%',
        'registered_rule': 'target_tolerance = max(1.0, radius_v/5.0) '
                           '(a10h_v22c_multi_scenario_dataset.py:66); '
                           'labels reused from val_hr_labels.csv',
        'nominal_rule': 'evaluator.target_tolerance := 1.0 override; '
                        'production split (B/2,B/4,B/4); adaptive '
                        '200->400; salt hr_t3',
        'files': {},
    }
    for p in (SELECT_CSV, LABEL_CKPT, OUT_CSV, PROV_MD):
        if os.path.exists(p):
            manifest['files'][os.path.basename(p)] = {
                'sha256': _sha256_file(p), 'bytes': os.path.getsize(p)}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f'[out] manifest -> {OUT_MANIFEST}')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', default='all',
                   choices=['select', 'labels', 'assemble', 'all'])
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage in ('select', 'all'):
        select_prefixes()
    if args.stage in ('labels', 'all'):
        stage_labels(args.workers)
    if args.stage in ('assemble', 'all'):
        stage_assemble()
    return 0


if __name__ == '__main__':
    sys.exit(main())
