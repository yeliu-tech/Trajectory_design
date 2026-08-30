# -*- coding: utf-8 -*-
"""sse_expD_highbudget_relabel.py — Exp D：high-budget (3200) J* relabel of aliased pairs。

任务书：state_sufficiency_extension / Exp D。
问题：Figure 3 的 aliasing heavy tail 是来自 state ambiguity，还是 production
（adaptive 200→400）completion-search 的有限预算噪声？

方法：对 Exp A full-25D 配对中的 high-aliasing（|ΔJ*_prod|>0.5σ_s）与等量
low-aliasing 对照（<0.25σ_s），以及 10D 诊断配对的敏感性集，用冻结 completion
协议在 budget=3200 下重算 J*（与 exp26 完全同实现：budget_split(3200)=
(1600,800,800)、nsga_mode='nested'、content_seed salt='hr_exp2'、CRN 升档嵌套）。

样本规则（固定，seed=20260829，不使用任何高预算结果）：
  - high 集：full-25D pairs 中 dJ_over_std>0.5，目标 200 对；
    按 (scenario, depth_band) 比例分配，stratum 内按 |ΔJ*| 排名等距取值
    （确定性，覆盖幅度全谱）。
  - low 对照：dJ_over_std<0.25，同规则同量（200 对）。
  - sensitivity：exp31 mechanism_pairs_raw.csv（10D 配对）中 dJ_over_std>0.5，
    同规则 150 对。
  - 三集去重得 unique prefixes；与 exp26 已有 3200 标签（同协议确定性结果）
    按 params_sha256 命中直接复用，不重算。

成本预估（exp26 实测：3200 档 median 396s / mean 438s）：
  ~900 unique × 440s ≈ 110 CPU-hours ≈ 11h @ 10 workers。

运行：
  python .../sse_expD_highbudget_relabel.py --stage select     # 样本集
  python .../sse_expD_highbudget_relabel.py --stage pilot      # 10 前缀计时试点
  python .../sse_expD_highbudget_relabel.py --stage collect --workers 10
  python .../sse_expD_highbudget_relabel.py --stage analyze    # 分析+图+表
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import exp2_relabel_train_val as e2  # noqa: E402
import exp6_independent_validation as e6  # noqa: E402
import hr_completion_oracle as orc  # noqa: E402

DATA_DIR = os.path.join(_PKG_DIR, 'data')
VAL_CSV = os.path.join(DATA_DIR, 'val_hr_labels.csv')
PARAMS_CACHE = os.path.join(DATA_DIR, 'prefix_rebuild_cache.csv')
EXPA_PAIRS = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension',
                          'expA_full25_aliasing', 'full25_aliasing_pairs.csv')
EXP31_PAIRS = os.path.join(_PKG_DIR, 'results',
                           'exp_targeted_correction_specificity',
                           'mechanism_pairs_raw.csv')
SCRATCH = os.environ.get('HCFTG_SCRATCH', 'I:/hcftg_scratch')
EXP26_CKPT = os.path.join(SCRATCH, 'exp26_label_convergence',
                          'exp26_labels_checkpoint.csv')
CKPT_DIR = os.path.join(SCRATCH, 'sse_expD_highbudget')
CKPT_CSV = os.path.join(CKPT_DIR, 'sse_expD_j3200_checkpoint.csv')

OUT_DIR = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension',
                       'expD_highbudget_relabel')

BUDGET = 3200
SEED_SALT = 'hr_exp2'
N_HIGH_PAIRS = 200
N_LOW_PAIRS = 200
N_SENS_PAIRS = 150
NOISE_BAND = 0.35
SEED = 20260829
STD_CUTS = (0.25, 0.5, 1.0)

CKPT_FIELDS = ['params_sha256', 'budget', 'seed', 'scenario_uid',
               'J_star_cont', 'evals_total', 'elapsed_sec', 'error']


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


# =============================================================================
# 数据加载（与 exp19/Exp A 完全一致的过滤，保证行索引约定相同）
# =============================================================================

def load_val():
    val = pd.read_csv(VAL_CSV)
    val = val[val['hr_rebuild_ok'] == 1].reset_index(drop=True)
    val = val[np.isfinite(val['hr_J_star_cont'])].reset_index(drop=True)
    return val


def depth_band(k):
    return pd.cut(k, [0, 3, 6, 8], labels=['shallow', 'mid', 'deep'])


def stratified_pick(pairs, n_target):
    """(scenario, depth_band) 比例分配；stratum 内按 |ΔJ*| 排名等距取值。"""
    d = pairs.copy()
    d['depth_band'] = depth_band(d['k_prefix_a'])
    d = d.sort_values('abs_dJ').reset_index(drop=True)
    counts = d.groupby(['scenario_uid_a', 'depth_band'], observed=True).size()
    alloc = (counts / counts.sum() * n_target)
    alloc = np.maximum(np.floor(alloc).astype(int), 1)
    # 调整总量到 n_target（按最大余数法去尾/补头）
    while alloc.sum() > n_target:
        i = int(np.argmax(alloc - counts.reindex(alloc.index).to_numpy()
                          / np.maximum(counts.reindex(alloc.index)
                                       .to_numpy(), 1)))
        alloc.iloc[i] -= 1
    take = []
    for (uid, band), n in alloc.items():
        g = d[(d['scenario_uid_a'] == uid) & (d['depth_band'] == band)]
        if len(g) <= n:
            take.extend(g.index.tolist())
            continue
        ranks = np.linspace(0, len(g) - 1, num=n).round().astype(int)
        take.extend(g.index[ranks].tolist())
    take = sorted(set(take))
    if len(take) > n_target:  # set 去重后仍超，按 |ΔJ*| 排名全局等距截断
        d2 = d.loc[take].sort_values('abs_dJ')
        idx = np.linspace(0, len(d2) - 1, num=n_target).round().astype(int)
        take = sorted(d2.index[idx].tolist())
    return d.loc[take].sort_index()


# =============================================================================
# stage select
# =============================================================================

def stage_select(val):
    os.makedirs(OUT_DIR, exist_ok=True)
    p25 = pd.read_csv(EXPA_PAIRS)
    high = p25[p25['dJ_over_std'] > 0.5]
    low = p25[p25['dJ_over_std'] < 0.25]
    log(f'[select] full25D pairs={len(p25)} high(>0.5σ)={len(high)} '
        f'low(<0.25σ)={len(low)}')
    high_s = stratified_pick(high, min(N_HIGH_PAIRS, len(high)))
    low_s = stratified_pick(low, min(N_LOW_PAIRS, len(low)))
    high_s = high_s.assign(set='high25')
    low_s = low_s.assign(set='low25')

    # 10D 敏感性集：先验证 exp31 行索引约定与 val 过滤一致
    p10 = pd.read_csv(EXP31_PAIRS)
    chk = p10.head(200)
    dj = np.abs(val.loc[chk['a_row'], 'hr_J_star_cont'].to_numpy()
                - val.loc[chk['b_row'], 'hr_J_star_cont'].to_numpy())
    assert np.allclose(dj, chk['abs_dJ'].to_numpy(), rtol=0, atol=1e-9), \
        'exp31 pairs 行索引约定与 val 过滤不一致，停止'
    sens_pool = p10[p10['dJ_over_std'] > 0.5]
    sens_s = stratified_pick(
        sens_pool.rename(columns={'abs_dJ': 'abs_dJ'}),
        min(N_SENS_PAIRS, len(sens_pool)))
    sens_s = sens_s.assign(set='sens10')

    # 统一 pair 结构
    def norm(df, src):
        out = df[['scenario_uid_a', 'k_prefix_a', 'a_row', 'b_row',
                  'pair_dist', 'abs_dJ', 'dJ_over_std', 'set']].copy()
        out['source'] = src
        out['a_J_star'] = val.loc[df['a_row'], 'hr_J_star_cont'].to_numpy()
        out['b_J_star'] = val.loc[df['b_row'], 'hr_J_star_cont'].to_numpy()
        out['a_sha'] = val.loc[df['a_row'], 'hr_params_sha256'].to_numpy()
        out['b_sha'] = val.loc[df['b_row'], 'hr_params_sha256'].to_numpy()
        return out
    pairs = pd.concat([norm(high_s, 'full25'), norm(low_s, 'full25'),
                       norm(sens_s, 'tenD')], ignore_index=True)
    pairs.insert(0, 'pair_id', [f'D_{i:04d}' for i in range(len(pairs))])
    pairs.to_csv(os.path.join(OUT_DIR, 'selected_pairs.csv'), index=False)

    # unique 前缀
    a = pairs[['scenario_uid_a', 'a_row', 'a_sha', 'set']].rename(
        columns={'scenario_uid_a': 'scenario_uid', 'a_row': 'row',
                 'a_sha': 'params_sha256'})
    b = pairs[['scenario_uid_a', 'b_row', 'b_sha', 'set']].rename(
        columns={'scenario_uid_a': 'scenario_uid', 'b_row': 'row',
                 'b_sha': 'params_sha256'})
    up = pd.concat([a, b]).drop_duplicates('params_sha256').reset_index(drop=True)
    meta = val[['hr_params_sha256', 'k_prefix', 'hr_J_star_cont',
                'difficulty']].rename(columns={'hr_params_sha256':
                                               'params_sha256'})
    up = up.merge(meta, on='params_sha256', how='left')
    sets = (pd.concat([a, b]).groupby('params_sha256')['set']
            .apply(lambda s: '|'.join(sorted(set(s)))))
    up['sets'] = up['params_sha256'].map(sets)
    up.to_csv(os.path.join(OUT_DIR, 'unique_prefixes.csv'), index=False)
    log(f'[select] pairs={len(pairs)} (high={len(high_s)} low={len(low_s)} '
        f'sens={len(sens_s)}) unique_prefixes={len(up)}')
    by_set = pairs['set'].value_counts().to_dict()
    log(f'[select] pairs by set: {by_set}')


# =============================================================================
# stage collect（仿 exp26 逐字结构，budget=3200 单档）
# =============================================================================

_W = {}


def _worker_init():
    e6._worker_init()
    _W['pools'] = e2.load_scenario_pools()


def _eval_task(task):
    uid, psha, params_t = task
    exp1 = e6._W['exp1']
    sc = _W['pools'][uid]
    prefix = np.asarray(params_t, dtype=np.float64)
    row = {'params_sha256': psha, 'budget': BUDGET, 'seed': 0,
           'scenario_uid': uid, 'J_star_cont': '', 'evals_total': 0,
           'elapsed_sec': 0.0, 'error': ''}
    try:
        seed, _ = orc.content_seed(uid, prefix, salt=SEED_SALT)
        row['seed'] = int(seed)
        t0 = time.time()
        if len(prefix) // 3 >= exp1.N_SEG:
            counter = {'n': 0}
            ev, st = exp1._full_eval(sc, prefix, counter)
            comps = [(st, ev, 'self', np.zeros(0))] if st is not None else []
            recs = orc._completion_records(exp1, sc, prefix, comps,
                                           e6._W['review_fn'])
            agg = orc._aggregate(recs)
            agg.update({'evals_total': counter['n'], 'n_errors': 0})
        else:
            _recs, agg = orc.evaluate_prefix(
                exp1, sc, prefix, seed, *orc.budget_split(BUDGET),
                e6._W['review_fn'], nsga_mode='nested')
        row['J_star_cont'] = float(agg['J_star_cont'])
        row['evals_total'] = int(agg['evals_total'])
        row['elapsed_sec'] = round(time.time() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def load_params_map():
    pc = pd.read_csv(PARAMS_CACHE, usecols=['params_sha256', 'params_json'])
    return pc.set_index('params_sha256')['params_json'].to_dict()


def build_tasks(up, params_map, reuse_exp26=True):
    """返回 (tasks, reused_rows)。exp26 已有 3200 结果按 sha 复用。"""
    done = set()
    if os.path.exists(CKPT_CSV):
        with open(CKPT_CSV, newline='', encoding='utf-8') as f:
            done = {r['params_sha256'] for r in csv.DictReader(f)}
    reused = pd.DataFrame()
    exp26_shas = set()
    if reuse_exp26 and os.path.exists(EXP26_CKPT):
        e26 = pd.read_csv(EXP26_CKPT)
        e26 = e26[(e26['budget'] == BUDGET)
                  & (e26['error'].isna() | (e26['error'] == ''))]
        exp26_shas = set(e26['params_sha256'])
        hit = up[up['params_sha256'].isin(exp26_shas)]
        reused = e26[e26['params_sha256'].isin(set(hit['params_sha256']))]
    tasks = []
    for r in up.itertuples():
        psha = r.params_sha256
        if psha in done or psha in exp26_shas or psha not in params_map:
            continue
        params_t = tuple(float(v) for v in json.loads(params_map[psha]))
        tasks.append((r.scenario_uid, psha, params_t))
    return tasks, reused


def stage_collect(up, workers, limit=None):
    params_map = load_params_map()
    tasks, reused = build_tasks(up, params_map)
    if limit:
        tasks = tasks[:limit]
    log(f'[collect] unique={len(up)} 待跑={len(tasks)} '
        f'(exp26 复用={len(reused)}，checkpoint 已有见上)')
    os.makedirs(CKPT_DIR, exist_ok=True)
    new = not os.path.exists(CKPT_CSV)
    n = 0
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with open(CKPT_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CKPT_FIELDS)
        if new:
            w.writeheader()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=int(workers),
                                     initializer=_worker_init) as ex:
                futs = [ex.submit(_eval_task, t) for t in tasks]
                for fut in as_completed(futs):
                    w.writerow(fut.result())
                    n += 1
                    if n % 10 == 0 or n == len(tasks):
                        f.flush()
                        el = time.time() - t0
                        eta = el / n * (len(tasks) - n) / 3600
                        log(f'[collect] {n}/{len(tasks)} '
                            f'elapsed={el/3600:.2f}h ETA={eta:.2f}h')
        else:
            _worker_init()
            for t in tasks:
                r = _eval_task(t)
                w.writerow(r)
                n += 1
                f.flush()
                log(f'[collect] {n}/{len(tasks)} '
                    f'elapsed={r["elapsed_sec"]}s')
    log('[collect] done')


# =============================================================================
# stage analyze
# =============================================================================

def boot_ci(vals, codes, stat, n_boot=1000, seed=2024):
    vals = np.asarray(vals, dtype=float)
    uniq = np.unique(codes)
    by = [vals[codes == u] for u in uniq]
    point = float(stat(vals))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b_ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        boots[b_] = stat(np.concatenate([by[i] for i in pick]))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def stage_analyze(up):
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = pd.read_csv(os.path.join(OUT_DIR, 'selected_pairs.csv'))
    val = load_val()

    ck = pd.read_csv(CKPT_CSV)
    ck = ck[ck['error'].isna() | (ck['error'] == '')]
    tasks, reused = build_tasks(up, load_params_map())
    j3200 = pd.concat([ck, reused[CKPT_FIELDS]], ignore_index=True)
    j3200 = j3200.drop_duplicates('params_sha256')
    j3200.to_csv(os.path.join(OUT_DIR, 'Jstar_3200_raw.csv'), index=False,
                 float_format='%.8f')

    # 锚点：与 exp26 重叠前缀的 3200 值逐字一致（CRN 确定性）
    anchor = {'n_ckpt_rows': int(len(ck)), 'n_reused_exp26': int(len(reused))}
    if len(reused):
        m = ck.merge(reused, on='params_sha256', suffixes=('_new', '_e26'))
        if len(m):
            same = np.allclose(m['J_star_cont_new'].astype(float),
                               m['J_star_cont_e26'].astype(float),
                               rtol=0, atol=1e-9)
            anchor['exp26_overlap_exact_match'] = bool(same)
            anchor['exp26_overlap_n'] = int(len(m))
            assert same, 'exp26 重叠前缀 3200 值不一致——确定性锚点失败'

    jmap = j3200.set_index('params_sha256')['J_star_cont'].astype(float)
    pairs['a_J3200'] = pairs['a_sha'].map(jmap)
    pairs['b_J3200'] = pairs['b_sha'].map(jmap)
    ok = pairs.dropna(subset=['a_J3200', 'b_J3200']).copy()
    ok['abs_dJ_3200'] = (ok['a_J3200'] - ok['b_J3200']).abs()

    # σ：主口径 = 场景内全部 val 前缀 production J* std（与诊断一致）；
    # sensitivity = 场景内被选中前缀的 J*3200 std（≥5 个才用，否则回退主口径）
    std_prod = val.groupby('scenario_uid')['hr_J_star_cont'].std(ddof=0)
    ok['sigma_prod'] = ok['scenario_uid_a'].map(std_prod)
    j3200_df = j3200.reset_index()
    up2 = up[['params_sha256', 'scenario_uid']]
    j3200_df = (j3200_df.drop(columns=['scenario_uid'], errors='ignore')
                .merge(up2, on='params_sha256', how='left'))
    std_3200 = (j3200_df.groupby('scenario_uid')['J_star_cont']
                .agg(['std', 'count']))
    std_3200_map = {u: (r['std'] if r['count'] >= 5 and r['std'] > 0 else np.nan)
                    for u, r in std_3200.iterrows()}
    ok['sigma_3200'] = ok['scenario_uid_a'].map(std_3200_map)
    ok['sigma_3200'] = ok['sigma_3200'].fillna(ok['sigma_prod'])
    ok['dJ3200_over_std'] = ok['abs_dJ_3200'] / ok['sigma_prod']
    ok['dJ3200_over_std3200'] = ok['abs_dJ_3200'] / ok['sigma_3200']
    ok['order_prod'] = np.sign(ok['a_J_star'] - ok['b_J_star'])
    ok['order_3200'] = np.sign(ok['a_J3200'] - ok['b_J3200'])
    ok['order_agree'] = ((ok['order_prod'] == ok['order_3200'])
                         | (ok['abs_dJ'] < 1e-9)).astype(float)
    ok.to_csv(os.path.join(OUT_DIR, 'highbudget_pair_comparison.csv'),
              index=False, float_format='%.8f')

    # ---- summary ----
    rows = []
    for setname, g in ok.groupby('set'):
        codes = pd.Categorical(g['scenario_uid_a']).codes
        row = {'set': setname, 'n_pairs': len(g),
               'n_scenarios': g['scenario_uid_a'].nunique()}
        pt, lo, hi = boot_ci(g['order_agree'].to_numpy(), codes,
                             lambda v: float(np.mean(v)))
        row.update({'order_agreement': pt, 'order_ci_lo': lo,
                    'order_ci_hi': hi})
        for c in STD_CUTS:
            pt, lo, hi = boot_ci(g['dJ3200_over_std'].to_numpy(), codes,
                                 lambda v, c=c: float((v > c).mean()))
            row[f'retain_{c}sigma'] = pt
            row[f'retain_{c}_ci_lo'] = lo
            row[f'retain_{c}_ci_hi'] = hi
            row[f'retain3200scale_{c}sigma'] = float(
                (g['dJ3200_over_std3200'] > c).mean())
        rows.append(row)
    # high vs low 的 3200 |ΔJ*|/σ 分布对比
    for setname in ('high25', 'low25'):
        g = ok[ok['set'] == setname]
        codes = pd.Categorical(g['scenario_uid_a']).codes
        pt, lo, hi = boot_ci(g['dJ3200_over_std'].to_numpy(), codes,
                             lambda v: float(np.median(v)))
        for r in rows:
            if r['set'] == setname:
                r['median_dJ3200_over_std'] = pt
                r['median_ci_lo'] = lo
                r['median_ci_hi'] = hi
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(OUT_DIR, 'highbudget_summary.csv'), index=False)

    # prefix 级 drift
    upj = up.merge(j3200[['params_sha256', 'J_star_cont']],
                   on='params_sha256', how='inner')
    upj['drift'] = (upj['J_star_cont'] - upj['hr_J_star_cont']).abs()
    from scipy.stats import spearmanr
    drift = {'n_prefixes': int(len(upj)),
             'median_abs_drift': float(upj['drift'].median()),
             'p90_abs_drift': float(upj['drift'].quantile(0.9)),
             'frac_drift_over_noise_band': float(
                 (upj['drift'] > NOISE_BAND).mean()),
             'spearman_prod_vs_3200': float(spearmanr(
                 upj['hr_J_star_cont'], upj['J_star_cont'])[0])}
    with open(os.path.join(OUT_DIR, 'highbudget_summary_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'anchor': anchor, 'prefix_drift': drift,
                   'budget': BUDGET, 'seed_select': SEED,
                   'noise_band': NOISE_BAND}, f, ensure_ascii=False, indent=2)
    log(f'[analyze] summary:\n{summ.to_string()}')
    log(f'[analyze] prefix drift: {drift}')
    draw_figure(ok, summ)
    log(f'[analyze done] -> {OUT_DIR}')


def draw_figure(ok, summ):
    # ---- 图 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    for setname, c in [('high25', '#b2182b'), ('low25', '#2166ac'),
                       ('sens10', '#666666')]:
        g = ok[ok['set'] == setname]
        ax.scatter(g['dJ_over_std'], g['dJ3200_over_std'], s=8, alpha=0.4,
                   color=c, label=setname, rasterized=True)
    lim = max(ok['dJ_over_std'].max(), ok['dJ3200_over_std'].max()) * 1.05
    ax.plot([0, lim], [0, lim], color='k', lw=0.8, alpha=0.4)
    ax.axhline(0.5, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(0.5, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('production |ΔJ*| / σ')
    ax.set_ylabel('3200-budget |ΔJ*| / σ')
    ax.set_title('(a) Pair |ΔJ*| under production vs 3200')
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.grid(alpha=0.25)

    ax = axes[1]
    labels, pts, los, his = [], [], [], []
    for _, r in summ.iterrows():
        labels.append(r['set'])
        pts.append(r['retain_0.5sigma'])
        los.append(r['retain_0.5sigma'] - r['retain_0.5_ci_lo'])
        his.append(r['retain_0.5_ci_hi'] - r['retain_0.5sigma'])
    ax.bar(range(len(pts)), pts, yerr=[los, his], capsize=4,
           color=['#b2182b', '#2166ac', '#666666'][:len(pts)], alpha=0.85,
           error_kw=dict(lw=1))
    ax.set_xticks(range(len(pts)), labels)
    ax.set_ylabel('P(|ΔJ*₃₂₀₀| > 0.5σ)')
    ax.set_title('(b) Aliasing retention at budget 3200')
    ax.grid(alpha=0.25, axis='y')

    ax = axes[2]
    for setname, c in [('high25', '#b2182b'), ('low25', '#2166ac')]:
        g = ok[ok['set'] == setname]
        v = np.sort(g['dJ3200_over_std'].to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.8,
                label=setname)
    for c0 in STD_CUTS:
        ax.axvline(c0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlim(0, max(1.6, float(np.quantile(
        ok['dJ3200_over_std'].to_numpy(), 0.99)) * 1.05))
    ax.set_xlabel('3200-budget |ΔJ*| / σ')
    ax.set_ylabel('ECDF')
    ax.set_title('(c) High vs low control at 3200')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_highbudget_aliasing.png'),
                dpi=600)
    fig.savefig(os.path.join(OUT_DIR, 'figure_highbudget_aliasing.pdf'))


def stage_anchor(up, n=2):
    """确定性锚点：抽 n 个 exp26 复用前缀现场重算，要求逐字一致。"""
    params_map = load_params_map()
    _tasks, reused = build_tasks(up, params_map)
    assert len(reused), '无 exp26 复用前缀，锚点不可用'
    pick = reused.head(n)
    _worker_init()
    rows = []
    for r in pick.itertuples():
        psha = r.params_sha256
        uid = up.loc[up['params_sha256'] == psha, 'scenario_uid'].iloc[0]
        params_t = tuple(float(v) for v in json.loads(params_map[psha]))
        new = _eval_task((uid, psha, params_t))
        same = bool(abs(float(new['J_star_cont'])
                        - float(r.J_star_cont)) < 1e-9)
        rows.append({'params_sha256': psha, 'scenario_uid': uid,
                     'J_exp26': float(r.J_star_cont),
                     'J_recomputed': float(new['J_star_cont']),
                     'exact_match': same,
                     'elapsed_sec': new['elapsed_sec']})
        log(f'[anchor] {psha[:12]} exp26={r.J_star_cont:.8f} '
            f'recomputed={new["J_star_cont"]:.8f} match={same}')
    res = {'n_checked': len(rows),
           'all_exact': bool(all(r['exact_match'] for r in rows)),
           'rows': rows}
    with open(os.path.join(OUT_DIR, 'anchor_recompute_check.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    assert res['all_exact'], 'exp26 复用前缀重算不一致——确定性锚点失败'
    log('[anchor] PASS')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['select', 'pilot', 'collect', 'analyze',
                             'anchor', 'figure'])
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    if args.stage == 'select':
        stage_select(load_val())
    elif args.stage == 'pilot':
        up = pd.read_csv(os.path.join(OUT_DIR, 'unique_prefixes.csv'))
        stage_collect(up, workers=1, limit=10)
    elif args.stage == 'collect':
        up = pd.read_csv(os.path.join(OUT_DIR, 'unique_prefixes.csv'))
        stage_collect(up, args.workers)
    elif args.stage == 'anchor':
        up = pd.read_csv(os.path.join(OUT_DIR, 'unique_prefixes.csv'))
        stage_anchor(up)
    elif args.stage == 'figure':
        ok = pd.read_csv(os.path.join(OUT_DIR,
                                      'highbudget_pair_comparison.csv'))
        summ = pd.read_csv(os.path.join(OUT_DIR, 'highbudget_summary.csv'))
        draw_figure(ok, summ)
    elif args.stage == 'analyze':
        up = pd.read_csv(os.path.join(OUT_DIR, 'unique_prefixes.csv'))
        stage_analyze(up)


if __name__ == '__main__':
    main()
