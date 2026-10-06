# -*- coding: utf-8 -*-
"""
t2_protocol_sensitivity.py — 审稿补充数据 T2：continuation-protocol
sensitivity（同一批 val prefix 在 P0/P1/P2 下的 J*）。

=====================================================================
登记（2026-08-29，运行前定稿；此后不改抽样规则、不改协议、不改预算、
不改输出 schema；结果如何都如实入报告）
=====================================================================

【目的】
  检查 J* 及 prefix 排序是否过度依赖当前 frozen continuation search
  protocol。P0 = 现有 production 标签（直接引用 val_hr_labels.csv，
  不重算）；P1/P2 只改策略分配，objective / constraints / target
  convention / adaptive 200→400 规则 / nested NSGA 模式全部不变。

【协议】
  P0：production（引用现有 hr_* 列；budget_split = (B/2, B/4, B/4)）
  P1：Sobol + multi-start SLSQP 主导：(n_sobol, n_slsqp, n_nsga)
      = (B/2, B/2, 0)；salt 'hr_t2p1'
  P2：evolutionary / NSGA-II 主导：(B/4, 0, 3B/4)；salt 'hr_t2p2'
  三者同 base 200、同近边界加倍 400（决策单元 = (scenario_uid,
  group_id)，leader gap ≤ 0.35 且 |active| ≥ 2，规则同 exp2）、同
  content_seed 方案（仅盐不同）。

【prefix 选择（冻结规则；不使用任何模型预测/误差）】
  1. 场景：val 180 池按 difficulty 分层，最大余数法分配 40 个名额，
     层内 scenario_uid 升序等距取样（systematic，起点 0）。
  2. 决策单元：每场景取 2 个完整 group（4 分支全保留）——
     unit_early = k_prefix 最小的 group（并列取 group_id 小者）；
     unit_late  = k_prefix 最大的 group（并列取 group_id 大者）；
     若同一 group 则 unit_early 改取 k 最小中 group_id 次小者。
  3. 规模：40 场景 × 2 单元 × 4 分支 = 320 prefixes（在 200–400 内）。

【输出】protocol_sensitivity/
  t2_selected_prefixes.csv    选择结果 + 选择规则字段
  t2_labels_checkpoint.csv    P1/P2 内容寻址标签 checkpoint（断点续跑）
  t2_protocol_results.csv     长表：prefix × protocol 的 J* / 五分量 /
                              budget / evals / best completion / 状态
  t2_manifest.json

用法：
  python t2_protocol_sensitivity.py --stage all --workers 64
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

# ---------------- T2 登记数值（冻结） ----------------
N_SCENARIOS = 40
UNITS_PER_SCENARIO = 2
BASE_BUDGET = 200
DOUBLE_BUDGET = 400
NOISE_BAND = 0.35
N_SEG = 9
SALTS = {'P1': 'hr_t2p1', 'P2': 'hr_t2p2'}


def protocol_split(protocol, budget):
    """各 protocol 的 (n_sobol, n_slsqp, n_nsga) 分配。"""
    b = int(budget)
    if protocol == 'P0':
        return b // 2, b // 4, b // 4
    if protocol == 'P1':
        return b // 2, b // 2, 0
    if protocol == 'P2':
        return b // 4, 0, (3 * b) // 4
    raise ValueError(protocol)


VAL_CSV = os.path.join(_REPO_ROOT, 'experiments',
                       'history_conditioned_ftg_final', 'data',
                       'val_hr_labels.csv')
OUT_DIR = os.environ.get('T2_OUT_DIR',
                         os.path.join(_RD_DIR, 'protocol_sensitivity'))
SELECT_CSV = os.path.join(OUT_DIR, 't2_selected_prefixes.csv')
LABEL_CKPT = os.path.join(OUT_DIR, 't2_labels_checkpoint.csv')
OUT_CSV = os.path.join(OUT_DIR, 't2_protocol_results.csv')
OUT_MANIFEST = os.path.join(OUT_DIR, 't2_manifest.json')

LABEL_FIELDS = (['protocol', 'params_sha256', 'budget', 'seed',
                 'scenario_uid', 'J_star_cont', 'cont_target',
                 'cont_safety', 'cont_dls', 'cont_constructability',
                 'cont_mechanical', 'J_star_old', 'R_k_eng',
                 'joint_pass_ratio', 'n_completions', 'n_feasible',
                 'any_feasible', 'evals_total', 'n_errors',
                 'best_method', 'best_tail_params', 'elapsed_sec'])


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
# 阶段 1：prefix 选择（冻结规则）
# =============================================================================

def select_prefixes():
    df = pd.read_csv(VAL_CSV)
    df = df[np.isfinite(df['hr_J_star_cont'])].copy()
    # ---- 场景分层等距抽样 ----
    scen = df.groupby('scenario_uid').agg(
        difficulty=('difficulty', 'first')).reset_index()
    strata = {d: g['scenario_uid'].sort_values().tolist()
              for d, g in scen.groupby('difficulty')}
    total = sum(len(v) for v in strata.values())
    # 最大余数法分配 40 个名额
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

    # ---- 每场景 2 个完整决策单元 ----
    rows = []
    for uid in picked:
        sub = df[df['scenario_uid'] == uid]
        g = sub.groupby('group_id')['k_prefix'].agg(['min', 'max', 'size'])
        g = g[g['size'] == 4]          # 只保留完整 4 分支单元
        kmin, kmax = int(g['min'].min()), int(g['max'].max())
        early = g[g['min'] == kmin].index.min()
        late = g[g['max'] == kmax].index.max()
        if early == late:
            cand = g[g['min'] == kmin].index.sort_values()
            early = cand[1] if len(cand) > 1 else cand[0]
        for role, gid in (('early', early), ('late', late)):
            unit = sub[sub['group_id'] == gid]
            for r in unit.itertuples(index=False):
                rows.append({
                    'scenario_uid': uid,
                    'difficulty': str(r.difficulty),
                    'group_id': int(gid),
                    'unit_role': role,
                    'branch_id': int(r.branch_id),
                    'k_prefix': int(r.k_prefix),
                    'n_seg_used': int(r.n_seg_used),
                    'hr_params_sha256': str(r.hr_params_sha256),
                    'P0_J_star_cont': float(r.hr_J_star_cont),
                    'P0_budget': float(r.hr_budget),
                    'P0_evals_total': float(r.hr_evals_total),
                    'P0_joint_pass_ratio': float(r.hr_joint_pass_ratio),
                    'P0_n_completions': float(r.hr_n_completions),
                    'P0_any_feasible': float(r.hr_any_feasible),
                })
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(SELECT_CSV, index=False)
    log(f'[select] {len(out)} prefixes / {out.scenario_uid.nunique()} 场景 '
        f'-> {SELECT_CSV}')
    return out


# =============================================================================
# 阶段 2：P1/P2 标签（production 同管线，仅策略分配与盐不同）
# =============================================================================

_W = {}


def _load_val_pools():
    """uid -> scenario dict（exp2.load_scenario_pools 的 val 部分）。"""
    pools = e2r.load_scenario_pools()
    return pools


def _worker_init():
    e2r._pin_repo_root()
    if _HCFTG_SCRIPTS not in sys.path:
        sys.path.insert(0, _HCFTG_SCRIPTS)
    import hr_completion_oracle as orc  # noqa: PLC0415
    exp1 = orc._import_module(orc.EXP1_SCRIPT, 'exp1_vp')
    from scripts.phase32.p32m_engineering_review import review_trajectory
    _W.update({'orc': orc, 'exp1': exp1, 'review_fn': review_trajectory,
               'pools': _load_val_pools()})


def _eval_one(task):
    """task = (protocol, uid, psha, budget, params_tuple)。"""
    protocol, uid, psha, budget, params_t = task
    orc, exp1 = _W['orc'], _W['exp1']
    sc = _W['pools'][uid]
    prefix = np.asarray(params_t, dtype=np.float64)
    seed, _ = orc.content_seed(uid, prefix, salt=SALTS[protocol])
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
            exp1, sc, prefix, seed, *protocol_split(protocol, budget),
            _W['review_fn'], nsga_mode='nested')

    row = {
        'protocol': protocol,
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
        'best_method': '',
        'best_tail_params': '',
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
        row['best_method'] = str(b['method'])
        row['best_tail_params'] = b['tail_params']
    return row


def _done_label_keys():
    keys = set()
    if os.path.exists(LABEL_CKPT):
        with open(LABEL_CKPT, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                keys.add((r['protocol'], r['params_sha256'],
                          int(r['budget'])))
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
    pools = _load_val_pools()
    # ---- 重建前缀参数（exp2 配方，44 特征对账）----
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

    def submit_tier(ex, protocol, budget, only=None):
        tasks = []
        for psha, (uid, pt) in states.items():
            if only is not None and psha not in only:
                continue
            if (protocol, psha, int(budget)) not in done:
                tasks.append((protocol, uid, psha, int(budget), pt))
        log(f'[labels] {protocol} 档 {budget}: 待跑 {len(tasks)}')
        _run_tasks(tasks, ex)

    with ProcessPoolExecutor(max_workers=int(workers),
                             initializer=_worker_init) as ex:
        for protocol in ('P1', 'P2'):
            submit_tier(ex, protocol, BASE_BUDGET)
            labels = _load_labels()
            lab = labels[(labels['protocol'] == protocol)
                         & (labels['budget'] == BASE_BUDGET)]
            lab200 = lab.set_index('params_sha256')['J_star_cont']
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
            log(f'[labels] {protocol} 加倍状态 {len(double)} 个')
            submit_tier(ex, protocol, DOUBLE_BUDGET, only=double)
    log('[labels] 完成')
    return 0


# =============================================================================
# 阶段 3：装配长表
# =============================================================================

def stage_assemble():
    sel = pd.read_csv(SELECT_CSV)
    labels = _load_labels()
    records = []
    for r in sel.itertuples(index=False):
        psha = str(r.hr_params_sha256)
        base = {'scenario_uid': r.scenario_uid,
                'difficulty': r.difficulty,
                'group_id': int(r.group_id),
                'unit_role': r.unit_role,
                'branch_id': int(r.branch_id),
                'k_prefix': int(r.k_prefix),
                'params_sha256': psha}
        # P0（引用现有 production 标签）
        records.append({**base, 'protocol': 'P0',
                        'J_star_cont': r.P0_J_star_cont,
                        'budget': r.P0_budget,
                        'evals_total': r.P0_evals_total,
                        'joint_pass_ratio': r.P0_joint_pass_ratio,
                        'n_completions': r.P0_n_completions,
                        'any_feasible': r.P0_any_feasible,
                        'status': 'reused_production_label'})
        # P1/P2（最高可用档）
        for protocol in ('P1', 'P2'):
            sub = labels[(labels['protocol'] == protocol)
                         & (labels['params_sha256'] == psha)]
            if sub.empty:
                records.append({**base, 'protocol': protocol,
                                'status': 'missing'})
                continue
            b = sub.loc[sub['budget'].idxmax()]
            records.append({**base, 'protocol': protocol,
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
                            'best_method': b['best_method'],
                            'best_tail_params': b['best_tail_params'],
                            'status': 'ok'})
    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    log(f'[out] {len(out)} 行 -> {OUT_CSV}')

    manifest = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'T2 continuation-protocol sensitivity',
        'n_scenarios': int(sel.scenario_uid.nunique()),
        'n_prefixes': int(len(sel)),
        'selection_rule': '40 scenarios stratified-systematic by '
                          'difficulty; per scenario 2 complete units '
                          '(min-k early / max-k late), 4 branches each',
        'protocols': {
            'P0': 'production labels reused from val_hr_labels.csv '
                  '(no recompute); split (B/2,B/4,B/4)',
            'P1': {'split': '(B/2,B/2,0)', 'salt': SALTS['P1']},
            'P2': {'split': '(B/4,0,3B/4)', 'salt': SALTS['P2']},
        },
        'budget_rule': 'adaptive 200->400, noise band 0.35, decision unit '
                       '= (scenario_uid, group_id); nsga_mode=nested',
        'files': {},
    }
    for p in (SELECT_CSV, LABEL_CKPT, OUT_CSV):
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
