# -*- coding: utf-8 -*-
"""sse_expB_T_decomposition.py — Exp B：T decomposition（current8 vs history2）。

任务书：state_sufficiency_extension / Exp B。
问题：M1=M0+T 的提升来自
  A. task-relative representation engineering（current8：把 M0 中已隐含存在的
     endpoint/target 信息重表达为 target-relative 坐标）；
  B. genuine path-history information（history2：prefix_t1_min_ell /
     prefix_t1_passed，M0 中不存在的 prefix 历史）；
  C. 两者互补；D. 其他。

臂（唯一变量 = state columns；其余全部冻结）：
  B0 = M0                 25D   复用 models/exp21/model_m0.joblib
  B1 = M0 + current8      33D   新训
  B2 = M0 + history2      27D   新训
  B3 = M0 + fullT (= M1)  35D   复用 models/exp21/model_m1.joblib

训练：e2.train_ensemble 默认（HistGB K=5, max_iter=300, seed=2024），
train/val/J* 标签/shuffle 协议与 exp21 逐字相同。新模型落 models/sse/。
评估：exp21/exp31 协议——val180 预测 → build_cell_table → pairwise_accuracy /
scenario_norm_spearman（e2._agg_metrics + e21.bootstrap_rank_metrics），
cell_selection_table → top1_regret_mean/median、top3_recall（e3.metrics_with_ci）。
paired scenario bootstrap：B1−B0、B2−B0、B3−B0、B3−B1、B3−B2。

锚点：B0/B3 指标必须与 exp31 落盘 pure_arm_ranking_summary.csv 的
M0/BT 行逐字一致（容差 1e-9；exp31 已与 exp21 核对）。

输出：experiments/state_sufficiency_extension/expB_T_decomposition/
  summary.csv  paired_contrasts.csv  per_scenario_metrics.csv
  global_metrics.json  train_timing.json  anchor_check.json
  figure_T_decomposition.png

运行：
  python .../sse_expB_T_decomposition.py [--skip-train]
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
                       'expB_T_decomposition')

SEED = 2024
N_BOOT = 1000

# 任务书定义（current8 / history2 的顺序按任务书登记）
CURRENT8 = ['bearing_rel_sin', 'bearing_rel_cos', 'off_target_angle_deg',
            'off_target1_angle_deg', 't1_dx', 't1_dy', 't1_dz', 't1_dist']
HISTORY2 = ['prefix_t1_min_ell', 'prefix_t1_passed']
assert set(CURRENT8 + HISTORY2) == {
    'bearing_rel_sin', 'bearing_rel_cos', 'off_target_angle_deg',
    't1_dx', 't1_dy', 't1_dz', 't1_dist', 'off_target1_angle_deg',
    'prefix_t1_min_ell', 'prefix_t1_passed'}, 'current8+history2 != T 组'

ARM_DIMS = {'B0': 25, 'B1': 33, 'B2': 27, 'B3': 35}
ARM_MODELS = {'B0': os.path.join(MODELS21, 'model_m0.joblib'),
              'B1': os.path.join(MODELS_SSE, 'model_b1.joblib'),
              'B2': os.path.join(MODELS_SSE, 'model_b2.joblib'),
              'B3': os.path.join(MODELS21, 'model_m1.joblib')}


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def build_arm_features():
    """列序与 exp21 一致：FEATURES_C  canonical 序内取 base|extra。"""
    base = list(e2.FEATURES_B)
    keep = lambda extra: [c for c in e2.FEATURES_C
                          if c in set(base) | set(extra)]  # noqa: E731
    fm = {'B0': list(base),
          'B1': keep(CURRENT8),
          'B2': keep(HISTORY2),
          'B3': keep(CURRENT8 + HISTORY2)}
    for a, n in ARM_DIMS.items():
        assert len(fm[a]) == n, f'{a} 列数 {len(fm[a])} != {n}'
    # B3 必须与冻结 M1 同列序
    p = joblib.load(ARM_MODELS['B3'])
    assert fm['B3'] == list(p['features']), 'B3 列序与 model_m1 不一致'
    return fm


def paired_boot_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """逐场景配对差 mean(a-b) 的 cluster bootstrap CI（同 exp31）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    mm = d[idx].mean(axis=1)
    return float(np.mean(d)), float(np.percentile(mm, 2.5)), \
        float(np.percentile(mm, 97.5)), int(len(d))


def stage_train(feat_map):
    os.makedirs(MODELS_SSE, exist_ok=True)
    df_tr, _df_va = e21.load_data(TRAIN_CSV, VAL_CSV)
    timing = {}
    for name in ('B1', 'B2'):
        mp = ARM_MODELS[name]
        feats = feat_map[name]
        if os.path.exists(mp):
            log(f'[train] {name} 已存在，跳过（{mp}）')
            continue
        t0 = time.time()
        members = e2.train_ensemble(df_tr, feats, seed=e2.SEED,
                                    k=e2.ENSEMBLE_K, max_iter=e2.MAX_ITER)
        wall = time.time() - t0
        payload = {'members': members, 'features': list(feats),
                   'label': e2.LABEL,
                   'metadata': {'seed': e2.SEED, 'ensemble': e2.ENSEMBLE_K,
                                'max_iter': e2.MAX_ITER,
                                'train_csv': TRAIN_CSV,
                                'n_train_rows': int(len(df_tr)),
                                'tag': 'sse_expB_T_decomposition'}}
        joblib.dump(payload, mp)
        timing[name] = {'n_features': len(feats), 'train_sec': wall}
        log(f'[train] {name} done in {wall:.1f}s -> {mp}')
    with open(os.path.join(OUT_DIR, 'train_timing.json'), 'w',
              encoding='utf-8') as f:
        json.dump(timing, f, indent=2)


def evaluate():
    df_tr_unused, df_va = e21.load_data(TRAIN_CSV, VAL_CSV)
    del df_tr_unused
    arms = list(ARM_MODELS)
    preds = {}
    for arm in arms:
        payload = joblib.load(ARM_MODELS[arm])
        assert len(payload['features']) == ARM_DIMS[arm]
        X = df_va[payload['features']].to_numpy(dtype=float)
        preds[arm] = e2.predict_ensemble_mean(payload['members'], X)
        log(f'[eval] {arm} 预测完成')

    uids = np.array(sorted(df_va['scenario_uid'].unique()))
    df_ev2 = df_va[['scenario_uid', e2.K_COL, e2.LABEL]].copy()
    for m in arms:
        df_ev2[f'pred_{m}'] = preds[m]
    cells2 = e2.build_cell_table(df_ev2, {m: f'pred_{m}' for m in arms})
    obs_rank = e2._agg_metrics(cells2, arms)
    rank_ci = e21.bootstrap_rank_metrics(cells2, arms, uids,
                                         n_boot=N_BOOT, seed=SEED)

    df_ev3 = df_va[['scenario_uid', 'group_id', e2.K_COL, 'branch_id',
                    e2.LABEL]].copy()
    sel_ci, sel_cells = {}, {}
    for m in arms:
        ct = e3.cell_selection_table(df_ev3.assign(__s=preds[m]), '__s')
        sel_cells[m] = ct
        sel_ci[m] = e3.metrics_with_ci(ct, uids, n_boot=N_BOOT, seed=SEED)

    glob = {}
    for m in arms:
        glob[m] = {'dimension': ARM_DIMS[m],
                   'pairwise_accuracy': obs_rank[m]['pairwise_accuracy'],
                   'pairwise_accuracy_ci':
                       rank_ci[m]['pairwise_accuracy']['ci95'],
                   'scenario_norm_spearman':
                       obs_rank[m]['scenario_norm_spearman'],
                   'scenario_norm_spearman_ci':
                       rank_ci[m]['scenario_norm_spearman']['ci95'],
                   'top1_regret_mean': sel_ci[m]['top1_regret_mean'],
                   'top1_regret_mean_ci': [sel_ci[m]['top1_regret_mean__ci_lo'],
                                           sel_ci[m]['top1_regret_mean__ci_hi']],
                   'top1_regret_median': sel_ci[m]['top1_regret_median'],
                   'top3_recall': sel_ci[m]['top3_recall']}

    acc = {}
    for uid, g in cells2.groupby('scenario_uid'):
        acc[uid] = {a: (g[f'{a}__n_correct'].sum() / g[f'{a}__n_pairs'].sum()
                        if g[f'{a}__n_pairs'].sum() > 0 else np.nan)
                    for a in arms}
    per_rows = []
    for m in arms:
        sct = sel_cells[m].groupby('scenario_uid').agg(
            top1_regret=('top1_regret', 'mean'),
            top3_recall=('top3_recall', 'mean'))
        for uid in uids:
            per_rows.append({'scenario_uid': uid, 'arm': m,
                             'pairwise_acc': acc[uid][m],
                             'top1_regret': float(sct.loc[uid, 'top1_regret'])
                             if uid in sct.index else np.nan,
                             'top3_recall': float(sct.loc[uid, 'top3_recall'])
                             if uid in sct.index else np.nan})
    per = pd.DataFrame(per_rows)
    per.to_csv(os.path.join(OUT_DIR, 'per_scenario_metrics.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'global_metrics.json'), 'w',
              encoding='utf-8') as f:
        json.dump(glob, f, indent=2)
    return per, glob


def stage_analyze(per, glob):
    # 锚点：B0/B3 vs exp31 落盘 M0/BT
    e31 = pd.read_csv(EXP31_SUMMARY).set_index('arm')
    anchor = {}
    for mine, ref in [('B0', 'M0'), ('B3', 'BT')]:
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
    for m in ARM_MODELS:
        g = glob[m]
        rows.append({'arm': m, 'dimension': g['dimension'],
                     'pairwise_accuracy': g['pairwise_accuracy'],
                     'accuracy_CI_low': g['pairwise_accuracy_ci'][0],
                     'accuracy_CI_high': g['pairwise_accuracy_ci'][1],
                     'scenario_norm_spearman': g['scenario_norm_spearman'],
                     'spearman_CI_low': g['scenario_norm_spearman_ci'][0],
                     'spearman_CI_high': g['scenario_norm_spearman_ci'][1],
                     'mean_top1_regret': g['top1_regret_mean'],
                     'regret_CI_low': g['top1_regret_mean_ci'][0],
                     'regret_CI_high': g['top1_regret_mean_ci'][1],
                     'median_top1_regret': g['top1_regret_median'],
                     'top3_recall': g['top3_recall']})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'summary.csv'),
                              index=False)

    pv = per.pivot(index='scenario_uid', columns='arm', values='top1_regret')
    pa = per.pivot(index='scenario_uid', columns='arm', values='pairwise_acc')
    contrasts = [('B1', 'B0'), ('B2', 'B0'), ('B3', 'B0'),
                 ('B3', 'B1'), ('B3', 'B2')]
    pr = []
    for hi, lo in contrasts:
        # Δregret = regret(lo) - regret(hi)：正值 = hi 更好
        d_reg, rlo_, rhi_, n = paired_boot_ci(pv[lo].to_numpy(),
                                              pv[hi].to_numpy())
        # Δacc = acc(hi) - acc(lo)：正值 = hi 更好
        d_acc, alo, ahi, _ = paired_boot_ci(pa[hi].to_numpy(),
                                            pa[lo].to_numpy())
        pr.append({'comparison': f'{hi} vs {lo}',
                   'delta_regret_pos_is_better': d_reg,
                   'regret_CI_low': rlo_, 'regret_CI_high': rhi_,
                   'delta_accuracy_pos_is_better': d_acc,
                   'accuracy_CI_low': alo, 'accuracy_CI_high': ahi,
                   'n_scenarios': n})
        log(f'[contrast] {hi} vs {lo}: Δregret={d_reg:+.4g} '
            f'[{rlo_:+.4g},{rhi_:+.4g}] Δacc={d_acc:+.4f} [{alo:+.4f},'
            f'{ahi:+.4f}]')
    pd.DataFrame(pr).to_csv(os.path.join(OUT_DIR, 'paired_contrasts.csv'),
                            index=False)

    # ---- 图 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = ['B0\nM0 (25D)', 'B1\nM0+current8 (33D)',
              'B2\nM0+history2 (27D)', 'B3\nM0+fullT=M1 (35D)']
    colors = ['#888888', '#4393c3', '#7b3294', '#2166ac']
    for ax, met, lo_k, hi_k, title in [
            (axes[0], 'pairwise_accuracy', 'accuracy_CI_low',
             'accuracy_CI_high', '(a) Pairwise ranking accuracy'),
            (axes[1], 'mean_top1_regret', 'regret_CI_low',
             'regret_CI_high', '(b) Mean top-1 regret')]:
        s = pd.DataFrame(rows)
        vals = s[met].to_numpy()
        lo = vals - s[lo_k].to_numpy()
        hi = s[hi_k].to_numpy() - vals
        ax.bar(range(4), vals, yerr=[lo, hi], capsize=4, color=colors,
               alpha=0.88, error_kw=dict(lw=1))
        for i, v in enumerate(vals):
            ax.text(i, v + hi[i], f'{v:.3f}' if v < 1 else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis='y')
        ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_T_decomposition.png'), dpi=600)
    fig.savefig(os.path.join(OUT_DIR, 'figure_T_decomposition.pdf'))
    log(f'[done] -> {OUT_DIR}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    feat_map = build_arm_features()
    if not args.skip_train:
        stage_train(feat_map)
    per, glob = evaluate()
    stage_analyze(per, glob)


if __name__ == '__main__':
    main()
