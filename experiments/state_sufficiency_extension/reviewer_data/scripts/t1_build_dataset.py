# -*- coding: utf-8 -*-
"""
t1_build_dataset.py — 审稿补充数据 T1：60 新场景的 prefix 候选池 +
production J* 标签 + 70D 特征，组装独立测试数据集。

=====================================================================
登记（2026-08-29，运行前定稿；此后不改抽样、不改协议、不改加倍规则、
不改输出 schema）
=====================================================================

【管线（全部复用冻结实现，仅换场景池与标签盐）】
  1. prefixes：experiments/segment_value_design/scripts/
     gen_state_outcome_data.py 主模式（与 v4_val_all.csv 同生成器、同
     参数：--groups-per-scenario 20、M=4 branches、K' completions +
     1 guided、seed-salt 0；group seed = sha256(uid|group_id)，新 uid
     ⇒ 新随机流），60 场景 × 20 组 × 4 分支 ≈ 4800 行。
  2. rebuild：exp2_relabel_train_val.rebuild_params 配方 1（main_new）
     重放前缀参数 + verify_row 44 特征对账（VERIFY_RTOL=1e-9；不通过
     记 rebuild_ok=0 并如实计数，不强行使用）。
  3. labels：hr_completion_oracle production 协议（与 val_hr_labels
     同一 evaluate_prefix + nsga_mode='nested'），基础预算 200 evals
     （budget_split 同 config/adaptive_budget.json），近边界加倍 400：
     决策单元 = (scenario_uid, group_id)，单元内 200 档 leader =
     J*_cont 最小者，gap ≤ NOISE_BAND(0.35) 且 |active| ≥ 2 的单元整档
     加倍；最终标签取最高可用档。seed = content_seed(uid, prefix,
     salt='hr_t1')（新盐登记；与 hr_exp2/hr_exp6/hr_exp26/expD 均不
     相同）。
  4. features：features_v3.build_state_features_v3(step_md=10.0) 全
     70 列（exp2_train_abc.FEATURES_C 口径，k_prefix 同 exp5/exp18
     公式 len(params)//3-1），内容寻址 checkpoint。
  5. assemble：t1_independent_test.csv（生成器行全列 + rebuild/label/
     feature 列）+ t1_manifest.json。

【断点续跑】labels / features 均为内容寻址 checkpoint，重跑自动跳过
  已完成键。

用法：
  python t1_build_dataset.py --stage all --workers 64
  分阶段：--stage prefixes|labels|features|assemble
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
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
from exp2_train_abc import FEATURES_C  # noqa: E402

# ---------------- T1 登记数值（冻结） ----------------
SEED_SALT_T1 = 'hr_t1'
BASE_BUDGET = 200
DOUBLE_BUDGET = 400
NOISE_BAND = 0.35
N_SEG = 9
GROUPS_PER_SCENARIO = 20

T1_DIR = os.environ.get(
    'T1_SCENARIO_DIR',
    os.path.join(_RD_DIR, 'independent_test', 't1_scenarios'))
T1_MANIFEST = os.path.join(T1_DIR, 'manifest.json')
OUT_DIR = os.environ.get('T1_OUT_DIR',
                         os.path.join(_RD_DIR, 'independent_test'))
PREFIX_CSV = os.path.join(OUT_DIR, 't1_prefix_rows.csv')
LABEL_CKPT = os.path.join(OUT_DIR, 't1_labels_checkpoint.csv')
FEAT_CKPT = os.path.join(OUT_DIR, 't1_features_checkpoint.csv')
OUT_CSV = os.path.join(OUT_DIR, 't1_independent_test.csv')
OUT_MANIFEST = os.path.join(OUT_DIR, 't1_manifest.json')

GEN_SCRIPT = os.path.join(
    _REPO_ROOT, 'experiments', 'segment_value_design', 'scripts',
    'gen_state_outcome_data.py')

LABEL_FIELDS = list(e2r.LABEL_FIELDS)
FEAT_FIELDS = ['scenario_uid', 'params_sha256'] + list(FEATURES_C)


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_t1_pools():
    """uid -> scenario dict（与 exp6.load_independent_pools 同链路）。"""
    e2r._pin_repo_root()
    from agent_common import load_scenarios
    scs = load_scenarios(T1_DIR, max_scenarios=None)
    return {str(sc.get('scenario_uid') or sc.get('name')): sc
            for sc in scs}


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
# 阶段 1：prefix 候选池生成（调冻结生成器）
# =============================================================================

def stage_prefixes(workers):
    cmd = [sys.executable, GEN_SCRIPT,
           '--scenario-dir', T1_DIR,
           '--scenario-loader', 'agent',
           '--groups-per-scenario', str(GROUPS_PER_SCENARIO),
           '--out', PREFIX_CSV,
           '--workers', str(int(workers)),
           '--resume']
    log('[prefixes] ' + ' '.join(cmd))
    r = subprocess.run(cmd, cwd=_REPO_ROOT)
    if r.returncode != 0:
        raise RuntimeError(f'gen_state_outcome_data 失败 rc={r.returncode}')
    df = pd.read_csv(PREFIX_CSV)
    log(f'[prefixes] {len(df)} 行 / {df.scenario_uid.nunique()} 场景')
    return 0


# =============================================================================
# 阶段 2/3 公用：重建 + 唯一状态集
# =============================================================================

def rebuild_rows(df, pools):
    """逐行重建前缀参数 + 44 特征对账。返回 {row_pos: rec}。"""
    out = {}
    n_fail = 0
    t0 = time.time()
    for pos, row in enumerate(df.itertuples(index=False)):
        rowd = row._asdict()
        uid = str(rowd['scenario_uid'])
        sc = pools.get(uid)
        rec = {'recipe': 'scenario_not_found', 'rebuild_ok': 0,
               'params_sha256': '', 'params_json': ''}
        if sc is not None:
            params, recipe = e2r.rebuild_params(rowd, sc)
            ok = params is not None and e2r.verify_row(rowd, sc, params)
            if not ok and params is not None:
                recipe = recipe + '_verify_fail'
            rec['recipe'] = recipe
            rec['rebuild_ok'] = int(bool(ok))
            if ok:
                rec['params_sha256'] = e2r._params_sha(params)
                rec['params_json'] = json.dumps([float(x) for x in params])
        n_fail += 1 - rec['rebuild_ok']
        out[pos] = rec
        if (pos + 1) % 1000 == 0 or pos + 1 == len(df):
            log(f'[rebuild] {pos + 1}/{len(df)} 失败 {n_fail} '
                f'({time.time() - t0:.0f}s)')
    if n_fail:
        log(f'[rebuild][WARN] 重建/对账失败 {n_fail} 行（如实剔除）')
    return out


def unique_states(df, rmap):
    """states: psha -> (uid, params_tuple)；row_meta: psha -> {(uid, gid)}。"""
    states, row_meta = {}, {}
    for pos, row in enumerate(df.itertuples(index=False)):
        rec = rmap[pos]
        if not rec['rebuild_ok']:
            continue
        psha = rec['params_sha256']
        if psha not in states:
            states[psha] = (str(row.scenario_uid),
                            tuple(json.loads(rec['params_json'])))
        row_meta.setdefault(psha, set()).add(
            (str(row.scenario_uid), int(row.group_id)))
    return states, row_meta


# =============================================================================
# 阶段 2：production J* 标签（200 全量 + 近边界加倍 400）
# =============================================================================

_W = {}


def _worker_init():
    e2r._pin_repo_root()
    if _HCFTG_SCRIPTS not in sys.path:
        sys.path.insert(0, _HCFTG_SCRIPTS)
    import hr_completion_oracle as orc  # noqa: PLC0415
    exp1 = orc._import_module(orc.EXP1_SCRIPT, 'exp1_vp')
    from scripts.phase32.p32m_engineering_review import review_trajectory
    _W.update({'orc': orc, 'exp1': exp1, 'review_fn': review_trajectory,
               'pools': load_t1_pools()})


def _eval_one(task):
    """task = (uid, params_sha, budget, params_tuple)。exp2._eval_one 同款，
    仅标签盐换 'hr_t1'。"""
    uid, psha, budget, params_t = task
    orc, exp1 = _W['orc'], _W['exp1']
    sc = _W['pools'][uid]
    prefix = np.asarray(params_t, dtype=np.float64)
    seed, _ = orc.content_seed(uid, prefix, salt=SEED_SALT_T1)
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
        if n_new % 500 == 0 or n_new == len(tasks):
            log(f'  [eval] {n_new}/{len(tasks)}')
    if buf:
        _append_rows(LABEL_CKPT, buf, LABEL_FIELDS)
    return n_new


def stage_labels(workers):
    df = pd.read_csv(PREFIX_CSV)
    pools = load_t1_pools()
    log(f'[labels] {len(df)} 行 / {len(pools)} 场景池')
    rmap = rebuild_rows(df, pools)
    states, row_meta = unique_states(df, rmap)
    log(f'[labels] 唯一状态 {len(states)} 个')

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
        # ---- 基础档 ----
        submit_tier(ex, BASE_BUDGET)
        # ---- 近边界加倍（决策单元 = (uid, group_id)，同 exp2 规则）----
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
        log(f'[labels] 加倍单元 '
            f'{sum(1 for u in unit_count if unit_count[u] >= 2)} 个，'
            f'加倍状态 {len(double)} 个')
        submit_tier(ex, DOUBLE_BUDGET, only=double)
    log('[labels] 完成')
    return 0


# =============================================================================
# 阶段 3：70D 特征（内容寻址 checkpoint）
# =============================================================================

def _feat_worker_init():
    e2r._pin_repo_root()
    if _HCFTG_SCRIPTS not in sys.path:
        sys.path.insert(0, _HCFTG_SCRIPTS)
    import features_v3  # noqa: PLC0415
    _W['fv3'] = features_v3
    _W['pools'] = load_t1_pools()


def _feat_task(task):
    """task = (uid, sha, params_tuple) -> 70 列特征行 dict（exp18 同款）。"""
    uid, sha, params_t = task
    sc = _W['pools'][uid]
    params = np.asarray(params_t, dtype=np.float64)
    row = _W['fv3'].build_state_features_v3(sc, params, step_md=10.0)
    row['k_prefix'] = float(len(params) // 3 - 1)
    out = {'scenario_uid': uid, 'params_sha256': sha}
    out.update({c: float(row[c]) for c in FEATURES_C})
    return out


def stage_features(workers):
    df = pd.read_csv(PREFIX_CSV)
    pools = load_t1_pools()
    rmap = rebuild_rows(df, pools)
    states, _ = unique_states(df, rmap)

    done = {}
    if os.path.exists(FEAT_CKPT):
        fdf = pd.read_csv(FEAT_CKPT)
        for r in fdf.itertuples(index=False):
            done[(r.scenario_uid, r.params_sha256)] = True
    log(f'[feat] checkpoint 已有 {len(done)} 条')
    tasks = []
    for psha, (uid, pt) in states.items():
        if (uid, psha) not in done:
            tasks.append((uid, psha, pt))
    log(f'[feat] 待算 {len(tasks)} 条 (workers={workers})')
    if tasks:
        buf, t_last, n = [], time.time(), 0
        with ProcessPoolExecutor(max_workers=int(workers),
                                 initializer=_feat_worker_init) as ex:
            futs = {ex.submit(_feat_task, t): t for t in tasks}
            for fut in as_completed(futs):
                buf.append(fut.result())
                n += 1
                if len(buf) >= 200 or (time.time() - t_last) >= 120:
                    _append_rows(FEAT_CKPT, buf, FEAT_FIELDS)
                    buf.clear()
                    t_last = time.time()
                if n % 1000 == 0 or n == len(tasks):
                    log(f'  [feat] {n}/{len(tasks)}')
        if buf:
            _append_rows(FEAT_CKPT, buf, FEAT_FIELDS)
    log('[feat] 完成')
    return 0


# =============================================================================
# 阶段 4：装配
# =============================================================================

def stage_assemble():
    df = pd.read_csv(PREFIX_CSV)
    pools = load_t1_pools()
    rmap = rebuild_rows(df, pools)
    states, _ = unique_states(df, rmap)

    labels = _load_labels()
    final = {}
    for psha in states:
        sub = labels[labels['params_sha256'] == psha]
        if sub.empty:
            continue
        best = sub.loc[sub['budget'].idxmax()]
        final[psha] = best.to_dict()

    feats = pd.read_csv(FEAT_CKPT).set_index(
        ['scenario_uid', 'params_sha256'])

    stratum = {}
    if os.path.exists(T1_MANIFEST):
        with open(T1_MANIFEST, 'r', encoding='utf-8') as f:
            for e in json.load(f)['scenarios']:
                stratum[e['scenario_uid']] = e['stratum']

    records, n_nolabel, n_rebuild_fail = [], 0, 0
    for pos, row in enumerate(df.itertuples(index=False)):
        rowd = row._asdict()
        rec = rmap[pos]
        uid = str(rowd['scenario_uid'])
        out = dict(rowd)
        out['stratum'] = stratum.get(uid, '')
        out['hr_rebuild_ok'] = int(rec['rebuild_ok'])
        out['hr_recipe'] = rec['recipe']
        out['hr_params_sha256'] = rec['params_sha256']
        n_rebuild_fail += 1 - int(rec['rebuild_ok'])
        lab = final.get(rec['params_sha256']) if rec['params_sha256'] else None
        if lab is None:
            n_nolabel += int(bool(rec['rebuild_ok']))
            lab = {}
        out['hr_seed'] = lab.get('seed', float('nan'))
        out['hr_budget'] = lab.get('budget', float('nan'))
        out['hr_J_star_cont'] = lab.get('J_star_cont', float('nan'))
        for c in ('cont_target', 'cont_safety', 'cont_dls',
                  'cont_constructability', 'cont_mechanical'):
            out['hr_' + c] = lab.get(c, float('nan'))
        out['hr_J_star_old'] = lab.get('J_star_old', float('nan'))
        out['hr_R_k_eng'] = lab.get('R_k_eng', float('nan'))
        out['hr_joint_pass_ratio'] = lab.get('joint_pass_ratio',
                                             float('nan'))
        out['hr_n_completions'] = lab.get('n_completions', float('nan'))
        out['hr_n_feasible'] = lab.get('n_feasible', float('nan'))
        out['hr_any_feasible'] = lab.get('any_feasible', float('nan'))
        out['hr_evals_total'] = lab.get('evals_total', float('nan'))
        out['hr_n_errors'] = lab.get('n_errors', float('nan'))
        # 70D 特征（特征列名与 FEATURES_C 一致，可能覆盖生成器同名列——
        # 口径同为 step_md=10 的 features_v3，数值一致）
        if rec['params_sha256']:
            key = (uid, rec['params_sha256'])
            if key in feats.index:
                fr = feats.loc[key]
                for c in FEATURES_C:
                    out[c] = float(fr[c])
        records.append(out)
    outdf = pd.DataFrame(records)
    outdf.to_csv(OUT_CSV, index=False)
    log(f'[out] {len(outdf)} 行（rebuild_fail={n_rebuild_fail}，'
        f'无标签={n_nolabel}）-> {OUT_CSV}')

    manifest = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'T1 independent test (reviewer data)',
        'n_rows': int(len(outdf)),
        'n_scenarios': int(outdf.scenario_uid.nunique()),
        'n_unique_states': int(len(states)),
        'n_rebuild_fail': int(n_rebuild_fail),
        'n_unlabeled_rebuild_ok': int(n_nolabel),
        'groups_per_scenario': GROUPS_PER_SCENARIO,
        'label_protocol': 'production adaptive 200->400, '
                          'nsga_mode=nested, budget_split from '
                          'config/adaptive_budget.json',
        'label_salt': SEED_SALT_T1,
        'noise_band': NOISE_BAND,
        'feature_recipe': 'features_v3.build_state_features_v3 '
                          '(step_md=10.0), FEATURES_C 70 cols',
        'files': {},
    }
    for p in (PREFIX_CSV, LABEL_CKPT, FEAT_CKPT, OUT_CSV, T1_MANIFEST):
        if os.path.exists(p):
            manifest['files'][os.path.basename(p)] = {
                'sha256': _sha256_file(p),
                'bytes': os.path.getsize(p)}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f'[out] manifest -> {OUT_MANIFEST}')
    return 0


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', default='all',
                   choices=['prefixes', 'labels', 'features', 'assemble',
                            'all'])
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage in ('prefixes', 'all'):
        stage_prefixes(args.workers)
    if args.stage in ('labels', 'all'):
        stage_labels(args.workers)
    if args.stage in ('features', 'all'):
        stage_features(args.workers)
    if args.stage in ('assemble', 'all'):
        stage_assemble()
    return 0


if __name__ == '__main__':
    sys.exit(main())
