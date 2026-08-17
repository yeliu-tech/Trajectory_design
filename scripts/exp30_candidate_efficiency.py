# -*- coding: utf-8 -*-
"""
exp30_candidate_efficiency.py — Candidate-assessment computational
efficiency（v15 论文补充实验；只新增测量，不改任何冻结对象）。

=====================================================================
协议要点（2026-08-17 主代理计划冻结；先审计后执行）
=====================================================================
- 审计发现：exp22 independent 池（36×16）为 step-9 children = 完整 9 段
  轨迹，生产协议对完整前缀的"补全"退化为 1 次自身评估
  （exp2_relabel_train_val._eval_one 的 N_SEG 分支），不能代表
  completion search 成本。故 Experiment A 改用含真实不完整前缀的
  冻结 val 池（exp25/exp28：30 场景 × step 1–8）。
- 子集规则（确定性落盘）：每 (scenario, step) 池按 hr_J_star_cont
  取 label-median 与 label-q75 两候选；经用户批准（2026-08-17）
  由 30 场景缩为 18 场景（6 个难度标签组各取 uid 排序前 3），
  即 18×8×2 = 288 prefixes（满足协议 ≥288 下限）。
  难度标签可核验：nw_retrain val 从场景名解析 easy/moderate/hard；
  nw_dual_val 读场景 JSON 的 difficulty_level。注意 simple/medium/
  strong 正式 stratum 只在 36 场景独立集注册，val 池无此注册。
- Direct completion = 冻结生产函数 hr_completion_oracle.evaluate_prefix
  （Sobol+SLSQP+NSGA-II nested CRN，内容 hash seed，salt='hr_exp2'），
  每 prefix 跑 200 与 400 两档；sequential（workers=1）。
  production 期望成本 = T200 + p_double × T400，p_double 由生产缓存
  data/checkpoints/exp2_labels.csv 的档位行数实测（不读其时间）。
- HGBR = models/exp21/model_m{0,1}.joblib（sha256 校验），5 成员
  HistGradientBoostingRegressor；score 口径 β=0（仅测延迟，不排序决策）。
- Experiment B = exp25+exp28 落盘预测的纯离线 K 曲线分析（无新模型运算）。
- Experiment C：NOT RUN（声明）。

输出目录：results/exp_efficiency/（新建，不覆盖任何已有文件）。

运行：
  python exp30_candidate_efficiency.py --smoke           # 烟测
  python exp30_candidate_efficiency.py                   # 全量（断点续跑）
  python exp30_candidate_efficiency.py --stage budget    # 单阶段
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp2_train_abc as e2  # noqa: E402  # train_ensemble 权威
import exp2_relabel_train_val as e2r  # noqa: E402  # SEED_SALT/pools
import exp5_beam_analysis as e5  # noqa: E402  # _prepare / FeatCache
import exp21_feature_ladder as e21  # noqa: E402  # feature map / load_data
import hr_completion_oracle as orc  # noqa: E402  # evaluate_prefix

SEED = 2024
N_BOOT = 1000
KS = [1, 2, 3, 4, 6, 8]
RANDOM_DRAWS = 100
FEAT_REPS = 20
INFER_REPS = 20
BATCH_REPS = 30
WARMUP = 20
AUDIT_REPS = 3          # 计时波动审计重复次数（tier=200）
AUDIT_PER_SCENARIO = 1  # 每场景 1 个（step=4 median 候选）

OUT_DIR = os.path.join(_PKG_DIR, 'results', 'exp_efficiency')
EXP25_CSV = os.path.join(_PKG_DIR, 'results', 'exp25_val_pool_predictions.csv')
EXP28_CSV = os.path.join(_PKG_DIR, 'results',
                         'exp28_val_pool_predictions_m0.csv')
PROD_LABELS = os.path.join(_PKG_DIR, 'data', 'checkpoints',
                           'exp2_labels.csv')
M0_PATH = os.path.join(_PKG_DIR, 'models', 'exp21', 'model_m0.joblib')
M1_PATH = os.path.join(_PKG_DIR, 'models', 'exp21', 'model_m1.joblib')
M0_SHA256 = ('1e93c4b2af36b201c1fe3c9ad85b0a17f12e180db008ab1e53a6bc'
             'dbfdbb8fe8')
M1_SHA256 = ('c3ca6bb3c0c4f4dd7e4f5b6f0a398f47ef10a2f97347d824d371e2'
             'fe285c2b9a')

# 与 well_trajectory_state_sufficiency_v15/figure_scripts_v15/figstyle_v15.py
# 逐字一致的配色（本脚本不跨目录 import，保持自包含）
C_M1 = '#0F4D92'
C_M0 = '#767676'
C_CTRL = '#D98E32'
C_DARK = '#333333'


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def _sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _ensure_out(smoke=False):
    d = os.path.join(OUT_DIR, 'smoke') if smoke else OUT_DIR
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# 子集选择（确定性；label 分位分层 + 难度分层；不使用 HGBR 预测误差）
# ---------------------------------------------------------------------------
_DUAL_JSON_DIR = os.path.join(_REPO_ROOT, 'experiments',
                              'segment_value_design', 'scenarios_dual',
                              'val')


def _subset_tier(uid):
    """validation-pool 场景的确定性难度标签（source:difficulty）。"""
    if uid.startswith('nw_dual_val'):
        with open(os.path.join(_DUAL_JSON_DIR, f'{uid}.json'),
                  encoding='utf-8') as f:
            return f"dual:{json.load(f)['difficulty_level']}"
    m = re.search(r'_(easy|moderate|hard)_', uid)
    assert m, f'无法从场景名解析难度: {uid}'
    return f'retrain:{m.group(1)}'


def select_subset(smoke=False):
    df = pd.read_csv(EXP25_CSV)
    df = df[df['step'].between(1, 8)].reset_index(drop=True)
    rows = []
    for (uid, step), g in df.groupby(['scenario_uid', 'step'], sort=True):
        g = g.sort_values('hr_J_star_cont', kind='stable').reset_index()
        n = len(g)
        for tag, q in (('median', 0.50), ('q75', 0.75)):
            pos = int(round(q * (n - 1)))
            r = g.iloc[pos]
            rows.append({'scenario_uid': uid, 'step': int(step),
                         'cand_idx': int(r['cand_idx']),
                         'params_sha256': r['params_sha256'],
                         'hr_J_star_cont': float(r['hr_J_star_cont']),
                         'pick': tag, 'pool_size': int(n),
                         'label_rank_pos': pos})
    sub = pd.DataFrame(rows)
    # 288 子集：6 个难度标签组各取 uid 排序前 3 个场景（18×8×2=288）
    sub['subset_tier'] = sub['scenario_uid'].map(_subset_tier)
    keep = []
    for _tier, g in sub.groupby('subset_tier', sort=True):
        keep.extend(sorted(g['scenario_uid'].unique())[:3])
    sub = sub[sub['scenario_uid'].isin(keep)].reset_index(drop=True)
    if smoke:
        uids = sorted(sub['scenario_uid'].unique())[:2]
        sub = sub[sub['scenario_uid'].isin(uids)]
        sub = sub[sub['step'].isin([1, 2])].reset_index(drop=True)
    return sub


# ---------------------------------------------------------------------------
# 参数复原（e5._prepare 确定性 rollout；结果缓存到 OUT 供断点复用）
# ---------------------------------------------------------------------------
def restore_params(sub, out_dir):
    cache = os.path.join(out_dir, 'params_cache.json')
    if os.path.exists(cache):
        with open(cache, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {uid: {sha: tuple(v) for sha, v in d.items()}
                for uid, d in data.items()}
    uids = sorted(sub['scenario_uid'].unique())
    log(f'[prepare] e5._prepare({len(uids)} uids, depth=9) ...')
    t0 = time.time()
    _rec, _steps, all_states = e5._prepare(uids, 9)
    log(f'[prepare] done in {(time.time() - t0) / 60:.1f} min')
    need = {}
    for uid in uids:
        shas = sub[sub['scenario_uid'] == uid]['params_sha256'].unique()
        need[uid] = {sha: list(all_states[uid][sha]) for sha in shas}
    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(need, f)
    return {uid: {sha: tuple(v) for sha, v in d.items()}
            for uid, d in need.items()}


def _load_models():
    assert _sha256(M0_PATH) == M0_SHA256, 'M0 sha256 漂移'
    assert _sha256(M1_PATH) == M1_SHA256, 'M1 sha256 漂移'
    loads = {}
    for name, path in (('M0', M0_PATH), ('M1', M1_PATH)):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            m = joblib.load(path)
            ts.append(time.perf_counter() - t0)
        loads[name] = {'model': m, 'load_times_s': ts,
                       'load_median_s': float(np.median(ts))}
    return loads


# ---------------------------------------------------------------------------
# Stage: hgbr —— 特征抽取 + 5 成员推理延迟 + batch 吞吐 + 模型加载
# ---------------------------------------------------------------------------
def stage_hgbr(sub, states, out_dir, smoke=False):
    import features_v3
    loads = _load_models()
    m0, m1 = loads['M0']['model'], loads['M1']['model']
    pools = e2r.load_scenario_pools()
    feat_reps = 5 if smoke else FEAT_REPS
    infer_reps = 5 if smoke else INFER_REPS
    batch_reps = 5 if smoke else BATCH_REPS

    # warm-up（不计入）
    log('[hgbr] warm-up ...')
    for r in sub.head(min(WARMUP, len(sub))).itertuples():
        sc = pools[r.scenario_uid]
        params = np.asarray(states[r.scenario_uid][r.params_sha256],
                            dtype=np.float64)
        row = features_v3.build_state_features_v3(sc, params, step_md=10.0)
        row['k_prefix'] = float(len(params) // 3 - 1)
        X = np.array([[row[c] for c in m1['features']]], dtype=float)
        for mem in m1['members']:
            mem.predict(X)

    rows, X0_all, X1_all = [], [], []
    log('[hgbr] per-candidate timing ...')
    for i, r in enumerate(sub.itertuples()):
        sc = pools[r.scenario_uid]
        params = np.asarray(states[r.scenario_uid][r.params_sha256],
                            dtype=np.float64)
        # feature extraction：重复 feat_reps 次取 median
        ft = []
        row = None
        for _ in range(feat_reps):
            t0 = time.perf_counter()
            row = features_v3.build_state_features_v3(sc, params,
                                                      step_md=10.0)
            row['k_prefix'] = float(len(params) // 3 - 1)
            ft.append(time.perf_counter() - t0)
        X1 = np.array([[row[c] for c in m1['features']]], dtype=float)
        X0 = np.array([[row[c] for c in m0['features']]], dtype=float)
        X0_all.append(X0[0])
        X1_all.append(X1[0])
        it = {}
        for name, mdl, X in (('m0', m0, X0), ('m1', m1, X1)):
            tt = []
            for _ in range(infer_reps):
                t0 = time.perf_counter()
                for mem in mdl['members']:
                    mem.predict(X)
                tt.append(time.perf_counter() - t0)
            it[name] = float(np.median(tt))
        rows.append({'scenario_uid': r.scenario_uid, 'step': int(r.step),
                     'cand_idx': int(r.cand_idx),
                     'params_sha256': r.params_sha256,
                     'feature_time_s': float(np.median(ft)),
                     'infer_m0_s': it['m0'], 'infer_m1_s': it['m1'],
                     'surrogate_m0_total_s': float(np.median(ft)) + it['m0'],
                     'surrogate_m1_total_s': float(np.median(ft)) + it['m1']})
        if (i + 1) % 50 == 0:
            log(f'  [hgbr] {i + 1}/{len(sub)}')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'hgbr_timing_raw.csv'), index=False)

    # batch throughput（5 成员整批前向）
    batch = {}
    for name, mdl, X in (('M0', m0, np.array(X0_all)),
                         ('M1', m1, np.array(X1_all))):
        tt = []
        for _ in range(batch_reps):
            t0 = time.perf_counter()
            for mem in mdl['members']:
                mem.predict(X)
            tt.append(time.perf_counter() - t0)
        med = float(np.median(tt))
        batch[name] = {'batch_wall_median_s': med,
                       'candidates_per_s': float(len(X) / med),
                       'n_candidates': int(len(X)), 'reps': batch_reps}
    meta = {'model_load': {k: {'load_median_s': v['load_median_s'],
                               'load_times_s': v['load_times_s']}
                           for k, v in loads.items()},
            'batch': batch,
            'reps': {'feature': feat_reps, 'inference': infer_reps,
                     'batch': batch_reps, 'warmup': WARMUP}}
    with open(os.path.join(out_dir, 'hgbr_timing_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    log('[hgbr] done')
    return df, meta


# ---------------------------------------------------------------------------
# Stage: train —— 同机重训冻结 M1 集成（仅取墙钟，不保存模型）
# ---------------------------------------------------------------------------
def stage_train(out_dir, smoke=False):
    if smoke:
        log('[train] smoke 模式跳过（全量运行才计时）')
        return None
    tp = os.path.join(out_dir, 'train_timing.json')
    if os.path.exists(tp):
        with open(tp, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        log(f"[train] 复用已有 train_timing.json "
            f"({meta['train_wall_s'] / 60:.1f} min)")
        return meta
    train_csv = os.path.join(_PKG_DIR, 'data', 'train_hr_labels.csv')
    val_csv = os.path.join(_PKG_DIR, 'data', 'val_hr_labels.csv')
    feat_map, _groups, _h45 = e21.build_feature_map()
    df_tr, _df_va = e21.load_data(train_csv, val_csv)
    log(f'[train] retrain frozen M1 ensemble '
        f'({len(df_tr)} rows, {len(feat_map["M1"])} feats) ...')
    t0 = time.perf_counter()
    members = e2.train_ensemble(df_tr, feat_map['M1'])
    wall = time.perf_counter() - t0
    meta = {'train_wall_s': float(wall),
            'n_train_rows': int(len(df_tr)),
            'n_train_scenarios': int(df_tr['scenario_uid'].nunique()),
            'n_features': int(len(feat_map['M1'])),
            'n_members': int(len(members)),
            'note': 'one-time offline training cost; 与在线推理成本分离'}
    with open(os.path.join(out_dir, 'train_timing.json'), 'w',
              encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    log(f'[train] done in {wall / 60:.1f} min')
    return meta


# ---------------------------------------------------------------------------
# Stage: direct —— 冻结 completion oracle 两档计时（sequential + checkpoint）
# ---------------------------------------------------------------------------
def stage_direct(sub, states, out_dir, smoke=False):
    exp1 = orc._import_module(orc.EXP1_SCRIPT, 'exp1_vp')
    from scripts.phase32.p32m_engineering_review import review_trajectory
    pools = e2r.load_scenario_pools()
    ckpt = os.path.join(out_dir, 'completion_timing_ckpt.csv')
    done = set()
    if os.path.exists(ckpt):
        d0 = pd.read_csv(ckpt)
        done = set(zip(d0['params_sha256'], d0['tier'], d0['rep']))
        log(f'[direct] resume: {len(done)} keys done')

    tasks = []
    for r in sub.itertuples():
        for tier in (200, 400):
            tasks.append((r, tier, 0))
    if not smoke:
        # 计时波动审计：每场景 step=4 median 候选，tier=200，重复 3 次
        for uid, g in sub.groupby('scenario_uid', sort=True):
            a = g[(g['step'] == 4) & (g['pick'] == 'median')]
            if len(a) == 0:
                a = g[g['pick'] == 'median'].head(1)
            for rep in range(1, AUDIT_REPS + 1):
                tasks.append((a.iloc[0], 200, rep))

    fieldnames = ['scenario_uid', 'step', 'cand_idx', 'params_sha256',
                  'pick', 'tier', 'rep', 'wall_s', 'cpu_s', 'evals_total',
                  'n_completions', 'J_star_cont', 'n_errors']
    new_rows = 0
    with open(ckpt, 'a', newline='', encoding='utf-8') as f:
        w = csv_writer(f, fieldnames, write_header=(not done))
        for i, (r, tier, rep) in enumerate(tasks):
            key = (r.params_sha256, tier, rep)
            if key in done:
                continue
            sc = pools[r.scenario_uid]
            prefix = np.asarray(states[r.scenario_uid][r.params_sha256],
                                dtype=np.float64)
            assert len(prefix) // 3 < e2r.N_SEG, \
                f'k=9 完整前缀不应进入子集: {r.params_sha256}'
            seed, _ = orc.content_seed(r.scenario_uid, prefix,
                                       salt=e2r.SEED_SALT)
            n_sobol, n_slsqp, n_nsga = orc.budget_split(tier)
            t0w, t0c = time.perf_counter(), time.process_time()
            _recs, agg = orc.evaluate_prefix(
                exp1, sc, prefix, seed, n_sobol, n_slsqp, n_nsga,
                review_trajectory, nsga_mode='nested')
            wall = time.perf_counter() - t0w
            cpu = time.process_time() - t0c
            w.writerow({'scenario_uid': r.scenario_uid, 'step': int(r.step),
                        'cand_idx': int(r.cand_idx),
                        'params_sha256': r.params_sha256, 'pick': r.pick,
                        'tier': int(tier), 'rep': int(rep),
                        'wall_s': round(wall, 4), 'cpu_s': round(cpu, 4),
                        'evals_total': int(agg['evals_total']),
                        'n_completions': int(agg['n_completions']),
                        'J_star_cont': float(agg['J_star_cont']),
                        'n_errors': int(agg.get('n_errors', 0))})
            f.flush()
            new_rows += 1
            if new_rows % 20 == 0:
                log(f'  [direct] +{new_rows} (task {i + 1}/{len(tasks)}) '
                    f'last wall={wall:.1f}s tier={tier}')
    log(f'[direct] done, new rows = {new_rows}')


def csv_writer(f, fieldnames, write_header=False):
    import csv as _csv
    w = _csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        w.writeheader()
    return w


# ---------------------------------------------------------------------------
# Stage: budget —— Experiment B（纯离线 K 曲线）
# ---------------------------------------------------------------------------
def stage_budget(out_dir):
    a = pd.read_csv(EXP25_CSV)
    b = pd.read_csv(EXP28_CSV)
    keys = ['scenario_uid', 'step', 'cand_idx', 'params_sha256']
    m = a.merge(b, on=keys, suffixes=('', '_m0chk'))
    assert len(m) == len(a) == len(b), 'exp25/exp28 池未精确对齐'
    assert np.allclose(m['hr_J_star_cont'], m['hr_J_star_cont_m0chk']), \
        '同一候选标签不一致'

    raw = []
    pools = list(m.groupby(['scenario_uid', 'step'], sort=True))
    for pi, ((uid, step), g) in enumerate(pools):
        y = g['hr_J_star_cont'].to_numpy(float)
        n = len(y)
        jbest = float(y.min())
        ystd = float(y.std(ddof=0))
        orders = {'M0': np.argsort(g['mu_m0'].to_numpy(float),
                                   kind='stable'),
                  'M1': np.argsort(g['mu_m1'].to_numpy(float),
                                   kind='stable')}
        rng = np.random.RandomState(SEED + pi)
        for k in KS:
            if k > n:
                continue
            for arm, o in orders.items():
                surv = float(bool(np.flatnonzero(o < k).__len__()
                                  and (y[o[:k]].min() <= jbest + 1e-12)))
                reg = float(y[o[:k]].min() - jbest)
                raw.append({'scenario_uid': uid, 'step': int(step),
                            'pool_size': n, 'K': k, 'arm': arm,
                            'survival': surv, 'retained_regret': reg,
                            'retained_regret_norm': reg / ystd
                            if ystd > 0 else np.nan})
            # random-K（池内有放回抽 subset 的无放回 K 集合，100 次）
            survs, regs = [], []
            for _ in range(RANDOM_DRAWS):
                idx = rng.choice(n, size=k, replace=False)
                survs.append(float(y[idx].min() <= jbest + 1e-12))
                regs.append(float(y[idx].min() - jbest))
            raw.append({'scenario_uid': uid, 'step': int(step),
                        'pool_size': n, 'K': k, 'arm': 'random',
                        'survival': float(np.mean(survs)),
                        'retained_regret': float(np.mean(regs)),
                        'retained_regret_norm': float(np.mean(regs) / ystd)
                        if ystd > 0 else np.nan})
    df = pd.DataFrame(raw)
    df.to_csv(os.path.join(out_dir, 'budget_tradeoff_raw.csv'), index=False)

    # 池 → 场景（步间平均）→ scenario-cluster bootstrap
    scen = df.groupby(['scenario_uid', 'K', 'arm'], sort=True)[
        ['survival', 'retained_regret', 'retained_regret_norm']].mean(
    ).reset_index()
    uids = np.array(sorted(scen['scenario_uid'].unique()))
    rng = np.random.RandomState(SEED)

    def _boot(values_by_uid):
        """values_by_uid: dict uid -> value；返回 (point, lo, hi)。"""
        pt = float(np.mean(list(values_by_uid.values())))
        reps = []
        for _ in range(N_BOOT):
            samp = rng.choice(uids, size=len(uids), replace=True)
            reps.append(float(np.mean([values_by_uid[u] for u in samp])))
        return pt, float(np.percentile(reps, 2.5)), \
            float(np.percentile(reps, 97.5))

    summ = []
    for k in KS:
        sub = scen[scen['K'] == k]
        if len(sub) == 0:
            continue
        arm_val = {}
        for arm in ('M0', 'M1', 'random'):
            v = sub[sub['arm'] == arm].set_index('scenario_uid')
            arm_val[arm] = v
            for metric in ('survival', 'retained_regret',
                           'retained_regret_norm'):
                p, lo, hi = _boot(v[metric].to_dict())
                summ.append({'K': k, 'arm': arm, 'metric': metric,
                             'point': p, 'ci_lo': lo, 'ci_hi': hi})
        d = (arm_val['M1'][['survival', 'retained_regret',
                            'retained_regret_norm']]
             - arm_val['M0'][['survival', 'retained_regret',
                              'retained_regret_norm']]).dropna()
        for metric in ('survival', 'retained_regret',
                       'retained_regret_norm'):
            p, lo, hi = _boot(d[metric].to_dict())
            summ.append({'K': k, 'arm': 'M1_minus_M0', 'metric': metric,
                         'point': p, 'ci_lo': lo, 'ci_hi': hi})
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(out_dir, 'budget_tradeoff_summary.csv'),
               index=False)
    log('[budget] done')
    return df, sdf


# ---------------------------------------------------------------------------
# Stage: analyze —— 汇总 A 的效率指标 + speedup CI + amortized cost
# ---------------------------------------------------------------------------
def stage_analyze(out_dir, smoke=False):
    direct = pd.read_csv(os.path.join(out_dir,
                                      'completion_timing_ckpt.csv'))
    hgbr = pd.read_csv(os.path.join(out_dir, 'hgbr_timing_raw.csv'))
    main = direct[direct['rep'] == 0]
    # 同一 (uid,step) 的 median/q75 选点可能命中同一候选（sha 相同）→
    # 同一前缀被独立测量两次。按 sha 聚合取均值（同计算的重复测量降噪）。
    t200 = (main[main['tier'] == 200]
            .groupby('params_sha256')
            .agg(wall_s=('wall_s', 'mean'),
                 evals_total=('evals_total', 'mean')))
    t400 = (main[main['tier'] == 400]
            .groupby('params_sha256')
            .agg(wall_s=('wall_s', 'mean'),
                 evals_total=('evals_total', 'mean')))

    lab = pd.read_csv(PROD_LABELS)
    n200 = int((lab['budget'] == 200).sum())
    n400 = int((lab['budget'] == 400).sum())
    p_double = n400 / n200

    rows = []
    # hgbr 侧同样按 sha 去重（同 sha 的重复选点取均值），避免重复行
    hgbr = (hgbr.groupby(['scenario_uid', 'step', 'params_sha256'],
                         as_index=False)
            .agg(surrogate_m1_total_s=('surrogate_m1_total_s', 'mean'),
                 surrogate_m0_total_s=('surrogate_m0_total_s', 'mean'),
                 feature_time_s=('feature_time_s', 'mean')))
    for r in hgbr.itertuples():
        sha = r.params_sha256
        if sha not in t200.index or sha not in t400.index:
            continue
        w200, w400 = float(t200.loc[sha, 'wall_s']), \
            float(t400.loc[sha, 'wall_s'])
        prod = w200 + p_double * w400
        rows.append({'scenario_uid': r.scenario_uid, 'step': int(r.step),
                     'params_sha256': sha,
                     'direct_200_wall_s': w200, 'direct_400_wall_s': w400,
                     'direct_200_evals': int(t200.loc[sha, 'evals_total']),
                     'direct_400_evals': int(t400.loc[sha, 'evals_total']),
                     'direct_production_expected_s': prod,
                     'surrogate_m1_total_s': float(r.surrogate_m1_total_s),
                     'surrogate_m0_total_s': float(r.surrogate_m0_total_s),
                     'feature_time_s': float(r.feature_time_s),
                     'speedup_m1': prod / float(r.surrogate_m1_total_s)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'candidate_timing_raw.csv'),
              index=False)

    uids = np.array(sorted(df['scenario_uid'].unique()))
    rng = np.random.RandomState(SEED)

    def _cluster_ci(stat_fn):
        reps = []
        by_uid = {u: df[df['scenario_uid'] == u] for u in uids}
        for _ in range(N_BOOT):
            samp = rng.choice(uids, size=len(uids), replace=True)
            d = pd.concat([by_uid[u] for u in samp], ignore_index=True)
            reps.append(stat_fn(d))
        return float(np.percentile(reps, 2.5)), \
            float(np.percentile(reps, 97.5))

    def _med(d):
        return float(d['speedup_m1'].median())

    def _geo(d):
        return float(np.exp(np.log(d['speedup_m1']).mean()))

    med_pt, geo_pt = _med(df), _geo(df)
    med_ci = _cluster_ci(_med)
    geo_ci = _cluster_ci(_geo)

    def _q(s, q):
        return float(s.quantile(q))

    summary = {
        'n_candidates': int(len(df)),
        'n_scenarios': int(len(uids)),
        'p_double_production': p_double,
        'p_double_source': ('data/checkpoints/exp2_labels.csv 档位行数 '
                            f'{n400}/{n200}（仅用计数，未用其计时）'),
        'direct_200': {'mean_s': float(df['direct_200_wall_s'].mean()),
                       'median_s': _q(df['direct_200_wall_s'], 0.5),
                       'iqr_s': [_q(df['direct_200_wall_s'], 0.25),
                                 _q(df['direct_200_wall_s'], 0.75)],
                       'p90_s': _q(df['direct_200_wall_s'], 0.9),
                       'mean_evals': float(df['direct_200_evals'].mean())},
        'direct_400': {'mean_s': float(df['direct_400_wall_s'].mean()),
                       'median_s': _q(df['direct_400_wall_s'], 0.5),
                       'iqr_s': [_q(df['direct_400_wall_s'], 0.25),
                                 _q(df['direct_400_wall_s'], 0.75)],
                       'p90_s': _q(df['direct_400_wall_s'], 0.9),
                       'mean_evals': float(df['direct_400_evals'].mean())},
        'direct_production_expected': {
            'mean_s': float(df['direct_production_expected_s'].mean()),
            'median_s': _q(df['direct_production_expected_s'], 0.5),
            'p90_s': _q(df['direct_production_expected_s'], 0.9)},
        'surrogate_m1': {
            'feature_median_ms': float(df['feature_time_s'].median() * 1e3),
            'inference_median_ms': float(
                (df['surrogate_m1_total_s'] - df['feature_time_s'])
                .median() * 1e3),
            'total_median_ms': float(df['surrogate_m1_total_s'].median()
                                     * 1e3)},
        'surrogate_m0': {
            'total_median_ms': float(df['surrogate_m0_total_s'].median()
                                     * 1e3)},
        'speedup_m1': {'median': med_pt, 'median_ci95': list(med_ci),
                       'geometric_mean': geo_pt,
                       'geometric_mean_ci95': list(geo_ci),
                       'bootstrap': 'scenario-cluster, B=1000, seed=2024'},
    }
    # 计时波动审计
    audit = direct[direct['rep'] > 0]
    if len(audit):
        av = audit.groupby('params_sha256')['wall_s'].agg(
            ['mean', 'std', 'count'])
        summary['timing_variability_audit'] = {
            'n_prefixes': int(len(av)), 'reps': AUDIT_REPS,
            'median_cv': float((av['std'] / av['mean']).median())}
    with open(os.path.join(out_dir, 'efficiency_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # amortized cost
    train_meta = None
    tp = os.path.join(out_dir, 'train_timing.json')
    if os.path.exists(tp):
        with open(tp, 'r', encoding='utf-8') as f:
            train_meta = json.load(f)
    c_train = train_meta['train_wall_s'] if train_meta else float('nan')
    c_direct = float(df['direct_production_expected_s'].median())
    c_online = float(df['surrogate_m1_total_s'].median())
    amort = []
    for n in (10, 50, 100, 500, 1000, 5000, 10000):
        amort.append({'N_candidates': n,
                      'C_direct_s': n * c_direct,
                      'C_surrogate_s': (c_train + n * c_online)
                      if train_meta else float('nan')})
    if train_meta and c_direct > c_online:
        breakeven = c_train / (c_direct - c_online)
    else:
        breakeven = float('nan')
    adf = pd.DataFrame(amort)
    adf['C_train_s'] = c_train
    adf['break_even_N'] = breakeven
    adf.to_csv(os.path.join(out_dir, 'amortized_cost.csv'), index=False)

    # candidate_timing_summary.csv（方法级汇总表）
    rows_s = [
        {'Method': 'Direct completion (tier 200)',
         'Median time/candidate (s)': summary['direct_200']['median_s'],
         'Mean time/candidate (s)': summary['direct_200']['mean_s'],
         'p90 (s)': summary['direct_200']['p90_s'],
         'Candidates/s': 1.0 / summary['direct_200']['median_s'],
         'Relative speedup': 1.0, 'One-time training cost (s)': ''},
        {'Method': 'Direct completion (tier 400)',
         'Median time/candidate (s)': summary['direct_400']['median_s'],
         'Mean time/candidate (s)': summary['direct_400']['mean_s'],
         'p90 (s)': summary['direct_400']['p90_s'],
         'Candidates/s': 1.0 / summary['direct_400']['median_s'],
         'Relative speedup': summary['direct_400']['median_s']
         / summary['direct_200']['median_s'],
         'One-time training cost (s)': ''},
        {'Method': 'Direct completion (production expected)',
         'Median time/candidate (s)':
             summary['direct_production_expected']['median_s'],
         'Mean time/candidate (s)':
             summary['direct_production_expected']['mean_s'],
         'p90 (s)': summary['direct_production_expected']['p90_s'],
         'Candidates/s': 1.0 / summary['direct_production_expected']
         ['median_s'],
         'Relative speedup': 1.0, 'One-time training cost (s)': ''},
        {'Method': 'HGBR surrogate M0 (feature + 5-member inference)',
         'Median time/candidate (s)':
             summary['surrogate_m0']['total_median_ms'] / 1e3,
         'Mean time/candidate (s)':
             float(df['surrogate_m0_total_s'].mean()),
         'p90 (s)': float(df['surrogate_m0_total_s'].quantile(0.9)),
         'Candidates/s': 1.0 / (summary['surrogate_m0']['total_median_ms']
                                / 1e3),
         'Relative speedup': c_direct /
             (summary['surrogate_m0']['total_median_ms'] / 1e3),
         'One-time training cost (s)': ''},
        {'Method': 'HGBR surrogate M1 (feature + 5-member inference)',
         'Median time/candidate (s)':
             summary['surrogate_m1']['total_median_ms'] / 1e3,
         'Mean time/candidate (s)':
             float(df['surrogate_m1_total_s'].mean()),
         'p90 (s)': float(df['surrogate_m1_total_s'].quantile(0.9)),
         'Candidates/s': 1.0 / (summary['surrogate_m1']['total_median_ms']
                                / 1e3),
         'Relative speedup': c_direct /
             (summary['surrogate_m1']['total_median_ms'] / 1e3),
         'One-time training cost (s)': c_train if train_meta else ''},
    ]
    pd.DataFrame(rows_s).to_csv(
        os.path.join(out_dir, 'candidate_timing_summary.csv'), index=False)
    log('[analyze] done')
    return summary


# ---------------------------------------------------------------------------
# Stage: figures
# ---------------------------------------------------------------------------
def stage_figures(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(out_dir, 'candidate_timing_raw.csv'))
    bs = pd.read_csv(os.path.join(out_dir, 'budget_tradeoff_summary.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.17,
                        wspace=0.30)

    # (a) latency，log scale
    ax = axes[0]
    series = [('Direct completion\n(production expected)',
               df['direct_production_expected_s'].to_numpy(), C_DARK),
              ('HGBR M0\n(feature + inference)',
               df['surrogate_m0_total_s'].to_numpy(), C_M0),
              ('HGBR M1\n(feature + inference)',
               df['surrogate_m1_total_s'].to_numpy(), C_M1)]
    for i, (lab, v, c) in enumerate(series):
        med = np.median(v)
        q25, q75 = np.percentile(v, [25, 75])
        ax.bar(i, med, width=0.55, color=c, zorder=3)
        ax.errorbar(i, med, yerr=[[med - q25], [q75 - med]], fmt='none',
                    ecolor='#222222', elinewidth=1.1, capsize=3, zorder=4)
        # 数值标签放在 IQR 误差棒上方，避免与 cap 重叠
        ax.text(i, q75 * 1.12,
                f'{med:.3g} s' if med >= 1 else f'{med * 1e3:.2f} ms',
                ha='center', va='bottom', fontsize=7.5, color=c)
    ax.set_yscale('log')
    # log 刻度用纯 ASCII 标签（默认 mathtext 的 U+2212 在当前字体缺字形）
    from matplotlib.ticker import FuncFormatter, FixedLocator
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: f'{v:g}'))
    ax.minorticks_off()
    ax.set_xticks(range(3))
    ax.set_xticklabels(['Direct\ncompletion', 'HGBR M0', 'HGBR M1'],
                       fontsize=7.5)
    ax.set_ylabel('wall time per candidate (s, log scale)', fontsize=8)
    ax.set_title('(a) Candidate-assessment latency', loc='left',
                 fontsize=9, fontweight='bold', pad=4)
    ax.grid(axis='y', color='#eeeeee', lw=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # (b) survival@K
    ax = axes[1]
    style = {'M0': (C_M0, 'o', 'M0 (conventional state)'),
             'M1': (C_M1, 's', 'M1 (+ target-approach)'),
             'random': (C_CTRL, '^', 'random-K')}
    sv = bs[bs['metric'] == 'survival']
    for arm, (c, mk, lab) in style.items():
        d = sv[sv['arm'] == arm].sort_values('K')
        ax.plot(d['K'], d['point'], marker=mk, ms=4, lw=1.4, color=c,
                label=lab, zorder=3)
        ax.fill_between(d['K'].to_numpy(float), d['ci_lo'], d['ci_hi'],
                        color=c, alpha=0.15, lw=0)
    ax.set_xlabel('retained candidates $K$', fontsize=8)
    ax.set_ylabel('true-best survival@$K$', fontsize=8)
    ax.set_xticks(KS)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc='lower right', frameon=False)
    ax.set_title('(b) Retained-budget vs ranking quality', loc='left',
                 fontsize=9, fontweight='bold', pad=4)
    ax.grid(color='#eeeeee', lw=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(
            out_dir, f'figure_efficiency.{ext}' if ext == 'pdf'
            else 'figure_efficiency_600dpi.png'),
            dpi=600 if ext == 'png' else None)
    plt.close(fig)

    # 补充图：retained regret vs K
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.17,
                        wspace=0.32)
    for j, (metric, ylab, ttl) in enumerate([
            ('retained_regret', 'retained-set regret $R_K$',
             '(a) Retained regret vs budget'),
            ('retained_regret_norm', 'normalized retained regret',
             '(b) Pool-std normalized')]):
        ax = axes[j]
        dd = bs[bs['metric'] == metric]
        for arm, (c, mk, lab) in style.items():
            d = dd[dd['arm'] == arm].sort_values('K')
            ax.plot(d['K'], d['point'], marker=mk, ms=4, lw=1.4, color=c,
                    label=lab, zorder=3)
            ax.fill_between(d['K'].to_numpy(float), d['ci_lo'], d['ci_hi'],
                            color=c, alpha=0.15, lw=0)
        ax.set_xlabel('retained candidates $K$', fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xticks(KS)
        ax.set_title(ttl, loc='left', fontsize=9, fontweight='bold', pad=4)
        ax.grid(color='#eeeeee', lw=0.5)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.legend(fontsize=7, loc='upper right', frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(
            out_dir, f'figure_budget_regret.{ext}' if ext == 'pdf'
            else 'figure_budget_regret_600dpi.png'),
            dpi=600 if ext == 'png' else None)
    plt.close(fig)
    log('[figures] done')


# ---------------------------------------------------------------------------
# Stage: manifest
# ---------------------------------------------------------------------------
def stage_manifest(out_dir, smoke=False):
    import sklearn
    mf = {
        'experiment': 'exp30_candidate_efficiency',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'smoke': bool(smoke),
        'environment': {'python': platform.python_version(),
                        'sklearn': sklearn.__version__,
                        'platform': platform.platform(),
                        'cpu_count': os.cpu_count()},
        'seeds': {'global': SEED, 'bootstrap': SEED,
                  'random_K': SEED, 'n_boot': N_BOOT},
        'parallelism': 'direct completion: sequential (workers=1); '
                       'HGBR inference/training: single process, '
                       'HistGB internal OpenMP threads',
        'frozen_inputs': {
            'exp25_val_pool_predictions.csv': _sha256(EXP25_CSV),
            'exp28_val_pool_predictions_m0.csv': _sha256(EXP28_CSV),
            'model_m0.joblib': M0_SHA256, 'model_m1.joblib': M1_SHA256},
        'subset_rule': ('30 val scenarios × steps 1-8 × label-median/q75 '
                        'picks (deterministic, no HGBR-error selection); '
                        'step-9 complete trajectories excluded (their '
                        '"completion" degenerates to a single self-'
                        'evaluation in the frozen production protocol)'),
        'direct_protocol': ('hr_completion_oracle.evaluate_prefix, nested '
                            'CRN, budget_split(200/400), content seed '
                            "salt='hr_exp2'; production expected = T200 + "
                            'p_double×T400, p_double from production tier '
                            'counts'),
    }
    with open(os.path.join(out_dir, 'run_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(mf, f, indent=2, ensure_ascii=False)
    log('[manifest] done')


STAGES = ['subset', 'budget', 'hgbr', 'train', 'direct', 'analyze',
          'figures', 'manifest']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stage', default='all', choices=STAGES + ['all'])
    args = ap.parse_args(argv)

    out_dir = _ensure_out(args.smoke)
    log(f'[exp30] out={out_dir} smoke={args.smoke} stage={args.stage}')
    t_all = time.time()

    sub_path = os.path.join(out_dir, 'subset_ids.csv')
    if args.stage in ('all', 'subset'):
        sub = select_subset(args.smoke)
        sub.to_csv(sub_path, index=False)
        log(f'[subset] {len(sub)} prefixes '
            f'({sub["scenario_uid"].nunique()} scenarios)')
    else:
        sub = pd.read_csv(sub_path)

    states = None
    if args.stage in ('all', 'hgbr', 'direct'):
        states = restore_params(sub, out_dir)

    if args.stage in ('all', 'budget'):
        stage_budget(out_dir)
    if args.stage in ('all', 'hgbr'):
        stage_hgbr(sub, states, out_dir, args.smoke)
    if args.stage in ('all', 'train'):
        stage_train(out_dir, args.smoke)
    if args.stage in ('all', 'direct'):
        stage_direct(sub, states, out_dir, args.smoke)
    if args.stage in ('all', 'analyze') and not args.smoke:
        stage_analyze(out_dir, args.smoke)
    if args.stage in ('all', 'figures') and not args.smoke:
        stage_figures(out_dir)
    if args.stage in ('all', 'manifest'):
        stage_manifest(out_dir, args.smoke)

    log(f'[exp30] total {(time.time() - t_all) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
