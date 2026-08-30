# -*- coding: utf-8 -*-
"""sse_expE_model_robustness.py — Exp E：model-family robustness。

任务书：state_sufficiency_extension / Exp E（第二优先级）。
问题：M1=M0+T 相对 M0 的增量价值是否只存在于 HGBR，
还是在归纳偏置不同的 learner 上方向一致？

模型族（M0 25D / M1 35D 同设置、不调参）：
  HGBR      —— 复用冻结 models/exp21/model_m0/m1.joblib（不重训）
  ExtraTrees—— sklearn ExtraTreesRegressor(n_estimators=100,
               random_state=seed+i)，其余 library default
  MLP       —— sklearn MLPRegressor(hidden_layer_sizes=(64,32),
               random_state=seed+i)，其余 library default；
               StandardScaler 仅用各成员训练子集 fit，val 仅 transform；
               M0/M1 相同处理逻辑

集成构造逐字仿 e2.train_ensemble：成员0=全量，成员 i≥1=scenario-cluster
bootstrap（RandomState(seed+1000i)，e2._cluster_bootstrap_indices）。
数据/标签/划分/评估协议与 exp21/exp31 逐字一致。

锚点：HGBR M0/M1 指标必须与 exp31 落盘 pure_arm_ranking_summary.csv
的 M0/BT 行一致（容差 1e-9）。

输出：experiments/state_sufficiency_extension/expE_model_robustness/
  summary.csv  paired_contrasts.csv  per_scenario_metrics.csv
  anchor_check.json  train_timing.json  figure_model_robustness.png

运行：
  python .../sse_expE_model_robustness.py [--skip-train]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import exp2_train_abc as e2  # noqa: E402
import exp3_conservative_ranking as e3  # noqa: E402
import exp21_feature_ladder as e21  # noqa: E402

TRAIN_CSV = os.path.join(_PKG_DIR, 'data', 'train_hr_labels.csv')
VAL_CSV = os.path.join(_PKG_DIR, 'data', 'val_hr_labels.csv')
MODELS21 = os.path.join(_PKG_DIR, 'models', 'exp21')
MODELS_SSE = os.path.join(_PKG_DIR, 'models', 'sse')
EXP31_SUMMARY = os.path.join(_PKG_DIR, 'results',
                             'exp_targeted_correction_specificity',
                             'pure_arm_ranking_summary.csv')
OUT_DIR = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension',
                       'expE_model_robustness')

SEED = 2024
N_BOOT = 1000
ET_TREES = 100
MLP_HIDDEN = (64, 32)

# (family, arm) -> model 路径；HGBR 复用冻结
ARMS = {}
for arm, mfile in [('M0', 'model_m0.joblib'), ('M1', 'model_m1.joblib')]:
    ARMS[('HGBR', arm)] = os.path.join(MODELS21, mfile)
for fam, tag in [('ExtraTrees', 'et'), ('MLP', 'mlp')]:
    for arm in ('M0', 'M1'):
        ARMS[(fam, arm)] = os.path.join(MODELS_SSE,
                                        f'model_e_{tag}_{arm.lower()}.joblib')


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def _make_member(fam, seed_i):
    if fam == 'ExtraTrees':
        return ('plain', ExtraTreesRegressor(n_estimators=ET_TREES,
                                             random_state=seed_i,
                                             n_jobs=-1))
    if fam == 'MLP':
        return ('scaled', MLPRegressor(hidden_layer_sizes=MLP_HIDDEN,
                                       random_state=seed_i))
    raise ValueError(fam)


def train_family_ensemble(df, feature_cols, fam, seed=SEED, k=e2.ENSEMBLE_K):
    """逐字仿 e2.train_ensemble 的成员构造；仅模型类不同。"""
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[e2.LABEL].to_numpy(dtype=float)
    members = []
    for i in range(int(k)):
        Xi, yi = X, y
        if i > 0:
            rng = np.random.RandomState(int(seed) + 1000 * i)
            idx = e2._cluster_bootstrap_indices(df, rng)
            Xi, yi = X[idx], y[idx]
        kind, m = _make_member(fam, int(seed) + i)
        if kind == 'scaled':
            sc = StandardScaler().fit(Xi)
            m.fit(sc.transform(Xi), yi)
            members.append({'scaler': sc, 'model': m})
        else:
            m.fit(Xi, yi)
            members.append(m)
        log(f'    [{fam}] member {i} done (n_train={len(Xi)})')
    return members


def predict_members(members, X):
    P = []
    for mem in members:
        if isinstance(mem, dict):
            P.append(mem['model'].predict(mem['scaler'].transform(X)))
        else:
            P.append(mem.predict(X))
    return np.column_stack(P).mean(axis=1)


def stage_train(feat_map):
    os.makedirs(MODELS_SSE, exist_ok=True)
    df_tr, _df_va = e21.load_data(TRAIN_CSV, VAL_CSV)
    timing = {}
    for fam in ('ExtraTrees', 'MLP'):
        for arm in ('M0', 'M1'):
            mp = ARMS[(fam, arm)]
            if os.path.exists(mp):
                log(f'[train] {fam}/{arm} 已存在，跳过')
                continue
            feats = feat_map[arm]
            t0 = time.time()
            members = train_family_ensemble(df_tr, feats, fam)
            wall = time.time() - t0
            joblib.dump({'members': members, 'features': list(feats),
                         'label': e2.LABEL,
                         'metadata': {'seed': SEED,
                                      'ensemble': e2.ENSEMBLE_K,
                                      'family': fam,
                                      'et_n_estimators':
                                          ET_TREES if fam == 'ExtraTrees'
                                          else None,
                                      'mlp_hidden':
                                          MLP_HIDDEN if fam == 'MLP'
                                          else None,
                                      'train_csv': TRAIN_CSV,
                                      'n_train_rows': int(len(df_tr)),
                                      'tag': 'sse_expE_model_robustness'}},
                        mp)
            timing[f'{fam}/{arm}'] = {'n_features': len(feats),
                                      'train_sec': wall}
            log(f'[train] {fam}/{arm} done in {wall:.1f}s -> {mp}')
    with open(os.path.join(OUT_DIR, 'train_timing.json'), 'w',
              encoding='utf-8') as f:
        json.dump(timing, f, indent=2)


def paired_boot_ci(a, b, n_boot=N_BOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    mm = d[idx].mean(axis=1)
    return float(np.mean(d)), float(np.percentile(mm, 2.5)), \
        float(np.percentile(mm, 97.5)), int(len(d))


def evaluate():
    _df_tr, df_va = e21.load_data(TRAIN_CSV, VAL_CSV)
    keys = list(ARMS)
    preds = {}
    for key in keys:
        payload = joblib.load(ARMS[key])
        X = df_va[payload['features']].to_numpy(dtype=float)
        if key[0] == 'HGBR':
            preds[key] = e2.predict_ensemble_mean(payload['members'], X)
        else:
            preds[key] = predict_members(payload['members'], X)
        log(f'[eval] {key[0]}/{key[1]} 预测完成')

    uids = np.array(sorted(df_va['scenario_uid'].unique()))
    df_ev2 = df_va[['scenario_uid', e2.K_COL, e2.LABEL]].copy()
    colmap = {}
    for key in keys:
        c = f'pred_{key[0]}_{key[1]}'
        df_ev2[c] = preds[key]
        colmap[f'{key[0]}_{key[1]}'] = c
    cells2 = e2.build_cell_table(df_ev2, colmap)
    obs_rank = e2._agg_metrics(cells2, list(colmap))
    rank_ci = e21.bootstrap_rank_metrics(cells2, list(colmap), uids,
                                         n_boot=N_BOOT, seed=SEED)

    df_ev3 = df_va[['scenario_uid', 'group_id', e2.K_COL, 'branch_id',
                    e2.LABEL]].copy()
    glob, sel_cells = {}, {}
    for key in keys:
        tag = f'{key[0]}_{key[1]}'
        ct = e3.cell_selection_table(df_ev3.assign(__s=preds[key]), '__s')
        sel_cells[tag] = ct
        ci = e3.metrics_with_ci(ct, uids, n_boot=N_BOOT, seed=SEED)
        glob[tag] = {'family': key[0], 'arm': key[1],
                     'pairwise_accuracy': obs_rank[tag]['pairwise_accuracy'],
                     'pairwise_accuracy_ci':
                         rank_ci[tag]['pairwise_accuracy']['ci95'],
                     'scenario_norm_spearman':
                         obs_rank[tag]['scenario_norm_spearman'],
                     'scenario_norm_spearman_ci':
                         rank_ci[tag]['scenario_norm_spearman']['ci95'],
                     'top1_regret_mean': ci['top1_regret_mean'],
                     'top1_regret_mean_ci': [ci['top1_regret_mean__ci_lo'],
                                             ci['top1_regret_mean__ci_hi']],
                     'top1_regret_median': ci['top1_regret_median']}

    acc = {}
    for uid, g in cells2.groupby('scenario_uid'):
        acc[uid] = {}
        for tag in colmap:
            npair = g[f'{tag}__n_pairs'].sum()
            acc[uid][tag] = (g[f'{tag}__n_correct'].sum() / npair
                             if npair > 0 else np.nan)
    per_rows = []
    for tag, ct in sel_cells.items():
        sct = ct.groupby('scenario_uid').agg(
            top1_regret=('top1_regret', 'mean'))
        for uid in uids:
            per_rows.append({'scenario_uid': uid, 'model': tag,
                             'pairwise_acc': acc[uid][tag],
                             'top1_regret': float(sct.loc[uid, 'top1_regret'])
                             if uid in sct.index else np.nan})
    per = pd.DataFrame(per_rows)
    per.to_csv(os.path.join(OUT_DIR, 'per_scenario_metrics.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'global_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(glob, f, indent=2)
    return per, glob


def stage_analyze(per, glob):
    # 锚点：HGBR M0/M1 vs exp31 落盘 M0/BT
    e31 = pd.read_csv(EXP31_SUMMARY).set_index('arm')
    anchor = {}
    for mine, ref in [('HGBR_M0', 'M0'), ('HGBR_M1', 'BT')]:
        da = abs(glob[mine]['pairwise_accuracy']
                 - float(e31.loc[ref, 'pairwise_accuracy']))
        dr = abs(glob[mine]['top1_regret_mean']
                 - float(e31.loc[ref, 'mean_top1_regret']))
        anchor[mine] = {'ref_arm': ref, 'abs_diff_accuracy': float(da),
                        'abs_diff_regret': float(dr),
                        'match': bool(da < 1e-9 and dr < 1e-9)}
        log(f'[anchor] {mine} vs exp31 {ref}: Δacc={da:.2e} Δreg={dr:.2e} '
            f'match={anchor[mine]["match"]}')
    with open(os.path.join(OUT_DIR, 'anchor_check.json'), 'w',
              encoding='utf-8') as f:
        json.dump(anchor, f, indent=2)
    assert all(v['match'] for v in anchor.values()), '锚点不一致，停止'

    rows = []
    for tag, g in glob.items():
        rows.append({'model': tag, 'family': g['family'], 'arm': g['arm'],
                     'pairwise_accuracy': g['pairwise_accuracy'],
                     'accuracy_CI_low': g['pairwise_accuracy_ci'][0],
                     'accuracy_CI_high': g['pairwise_accuracy_ci'][1],
                     'mean_top1_regret': g['top1_regret_mean'],
                     'regret_CI_low': g['top1_regret_mean_ci'][0],
                     'regret_CI_high': g['top1_regret_mean_ci'][1],
                     'median_top1_regret': g['top1_regret_median']})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'summary.csv'),
                              index=False)

    pv = per.pivot(index='scenario_uid', columns='model',
                   values='top1_regret')
    pa = per.pivot(index='scenario_uid', columns='model',
                   values='pairwise_acc')
    pr = []
    for fam in ('HGBR', 'ExtraTrees', 'MLP'):
        hi, lo = f'{fam}_M1', f'{fam}_M0'
        d_reg, rlo_, rhi_, n = paired_boot_ci(pv[lo].to_numpy(),
                                              pv[hi].to_numpy())
        d_acc, alo, ahi, _ = paired_boot_ci(pa[hi].to_numpy(),
                                            pa[lo].to_numpy())
        pr.append({'family': fam, 'comparison': f'{hi} vs {lo}',
                   'delta_regret_pos_is_better': d_reg,
                   'regret_CI_low': rlo_, 'regret_CI_high': rhi_,
                   'delta_accuracy_pos_is_better': d_acc,
                   'accuracy_CI_low': alo, 'accuracy_CI_high': ahi,
                   'n_scenarios': n})
        log(f'[contrast] {fam}: Δregret={d_reg:+.4g} [{rlo_:+.4g},'
            f'{rhi_:+.4g}] Δacc={d_acc:+.4f} [{alo:+.4f},{ahi:+.4f}]')
    pd.DataFrame(pr).to_csv(os.path.join(OUT_DIR, 'paired_contrasts.csv'),
                            index=False)
    draw_figure(pd.DataFrame(rows))
    log(f'[done] -> {OUT_DIR}')


def draw_figure(summ):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fams = ['HGBR', 'ExtraTrees', 'MLP']
    x = np.arange(len(fams))
    w = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for j, (arm, color) in enumerate([('M0', '#888888'), ('M1', '#2166ac')]):
        for ax, met, lo_c, hi_c, ttl in [
                (axes[0], 'pairwise_accuracy', 'accuracy_CI_low',
                 'accuracy_CI_high', '(a) Pairwise ranking accuracy'),
                (axes[1], 'mean_top1_regret', 'regret_CI_low',
                 'regret_CI_high', '(b) Mean top-1 regret')]:
            sub = summ[summ['arm'] == arm].set_index('family').loc[fams]
            vals = sub[met].to_numpy()
            lo = vals - sub[lo_c].to_numpy()
            hi = sub[hi_c].to_numpy() - vals
            ax.bar(x + (j - 0.5) * w, vals, w, yerr=[lo, hi], capsize=3,
                   color=color, alpha=0.88, error_kw=dict(lw=1),
                   label=arm if ax is axes[0] else None)
            for xi, v, h in zip(x + (j - 0.5) * w, vals, hi):
                ax.text(xi, v + h, f'{v:.3f}' if v < 1 else f'{v:.2f}',
                        ha='center', va='bottom', fontsize=7)
    for ax in axes:
        ax.set_xticks(x, fams)
        ax.grid(alpha=0.25, axis='y')
        ax.margins(y=0.18)
    axes[0].set_title('(a) Pairwise ranking accuracy')
    axes[1].set_title('(b) Mean top-1 regret')
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT_DIR, 'figure_model_robustness.png'),
                dpi=600)
    fig.savefig(os.path.join(OUT_DIR, 'figure_model_robustness.pdf'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.figure_only:
        draw_figure(pd.read_csv(os.path.join(OUT_DIR, 'summary.csv')))
        return
    m0 = list(joblib.load(ARMS[('HGBR', 'M0')])['features'])
    m1 = list(joblib.load(ARMS[('HGBR', 'M1')])['features'])
    assert len(m0) == 25 and len(m1) == 35
    feat_map = {'M0': m0, 'M1': m1}
    if not args.skip_train:
        stage_train(feat_map)
    per, glob = evaluate()
    stage_analyze(per, glob)


if __name__ == '__main__':
    main()
