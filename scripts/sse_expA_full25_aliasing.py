# -*- coding: utf-8 -*-
"""sse_expA_full25_aliasing.py — Exp A：完整 25D M0 checkpoint-state aliasing audit。

任务书：state_sufficiency_extension / Exp A。
问题：原 Figure 3 诊断用 10 列匹配向量（gate4 冻结 MATCH_COLS），严格证明的是
"10D selected checkpoint descriptors 接近的 prefixes 可有不同 J*"。本实验把匹配
空间扩展到完整 25D M0（exp2_feature_groups_v3.json B 组），检验 heavy tail 是否
依然存在。

方法（逐字沿用 exp19 冻结口径，仅替换匹配列）：
  - 同 scenario、同 k_prefix 分组；
  - 场景内标准化（mean/std，ddof=0，std=0→1，常数列自动失效）；
  - 贪心不重复最近邻配对（exp19.greedy_nn_pairs，欧氏距离）；
  - 阈值 = 组内全部候选对距离中位数 × thr_frac，thr_frac ∈ {0.25, 0.5, 1.0}；
  - 统计单位 = scenario，scenario-cluster bootstrap B=1000 seed=2024。

注意：M0 25 列中 k_prefix + 10 场景常数在同 (uid,k) 组内恒定，标准化后零贡献，
有效匹配维 = 15 个即时量。本报告如实写明这一点。

锚点校验：同协议 10D 复算必须与 exp19 落盘（6,593 对 / median 0.01253 /
frac>0.5σ=0.12362）逐字一致。

输出（experiments/state_sufficiency_extension/expA_full25_aliasing/）：
  full25_aliasing_pairs.csv        （thr_frac=1.0 主配对，逐对）
  full25_aliasing_summary.csv      （10D vs 25D 并列，含 CI）
  full25_aliasing_sensitivity.csv  （3 档阈值 × 2 匹配空间）
  full25_aliasing_anchor.json      （锚点校验 + 运行元数据）
  figure_full25_aliasing.png       （600dpi）

运行：
  python experiments/history_conditioned_ftg_final/scripts/sse_expA_full25_aliasing.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import exp19_aliasing_audit as e19  # noqa: E402

VAL_CSV = os.path.join(_PKG_DIR, 'data', 'val_hr_labels.csv')
GROUPS_JSON = os.path.join(_PKG_DIR, 'results', 'exp2_feature_groups_v3.json')
EXP19_SUMMARY = os.path.join(_REPO, 'paper_ftg_package', 'revision_20260730',
                             'aliasing_audit_summary.json')
OUT_DIR = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension',
                       'expA_full25_aliasing')

LABEL = e19.LABEL                    # hr_J_star_cont
STD_CUTS = e19.STD_CUTS              # [0.25, 0.5, 1.0]
THR_FRACS = e19.THR_FRACS            # [0.25, 0.5, 1.0]
SEED = e19.SEED
N_BOOT = e19.N_BOOT

# exp19 落盘锚点（aliasing_audit_summary.json, analysis_7）
ANCHOR_10D = {'n_pairs': 6593,
              'median_dJ_over_std': 0.01253025576543161,
              'frac_over_0.5std': 0.12361595631730624}


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def load_val():
    val = pd.read_csv(VAL_CSV)
    val = val[val['hr_rebuild_ok'] == 1].reset_index(drop=True)
    val = val[np.isfinite(val[LABEL])].reset_index(drop=True)
    return val


def build_pairs_with_cols(val, match_cols, thr_frac=1.0):
    """逐字复用 exp19.build_within_pairs，仅替换其全局 MATCH_COLS。"""
    saved = e19.MATCH_COLS
    e19.MATCH_COLS = list(match_cols)
    try:
        df = e19.build_within_pairs(val, pool_across_k=False, thr_frac=thr_frac)
    finally:
        e19.MATCH_COLS = saved
    return df


def pair_metrics(df):
    """一组配对的核心指标 + scenario cluster bootstrap CI。"""
    djn = df['dJ_over_std'].to_numpy(dtype=float)
    codes = pd.Categorical(df['scenario_uid_a']).codes
    out = {'n_pairs': int(len(df)),
           'n_scenarios': int(df['scenario_uid_a'].nunique()),
           'pair_dist_median': float(df['pair_dist'].median())}
    pt, lo, hi = e19.boot_ci_within(djn, codes, lambda v: float(np.median(v)))
    out['median_dJ_over_std'] = {'point': pt, 'ci_lo': lo, 'ci_hi': hi}
    for c in STD_CUTS:
        pt, lo, hi = e19.boot_ci_within(djn, codes, e19.frac_over_cut(c))
        out[f'frac_over_{c}std'] = {'point': pt, 'ci_lo': lo, 'ci_hi': hi}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(GROUPS_JSON, encoding='utf-8') as f:
        m0_cols = json.load(f)['groups']['B_scenario_plus_immediate']['features']
    assert len(m0_cols) == 25
    log(f'[A] M0 25 cols loaded from exp2_feature_groups_v3.json')

    val = load_val()
    log(f'[A] val rows={len(val)} scenarios={val["scenario_uid"].nunique()}')
    missing = [c for c in m0_cols if c not in val.columns]
    assert not missing, f'missing cols: {missing}'

    anchor = {'anchor_10d_expected': ANCHOR_10D}

    # ---- 锚点：10D 复算必须逐字复现 exp19 ----
    p10 = build_pairs_with_cols(val, e19.MATCH_COLS, thr_frac=1.0)
    a10 = {'n_pairs': int(len(p10)),
           'median_dJ_over_std': float(p10['dJ_over_std'].median()),
           'frac_over_0.5std': float((p10['dJ_over_std'] > 0.5).mean())}
    a10['match'] = (a10['n_pairs'] == ANCHOR_10D['n_pairs']
                    and abs(a10['median_dJ_over_std']
                            - ANCHOR_10D['median_dJ_over_std']) < 1e-12
                    and abs(a10['frac_over_0.5std']
                            - ANCHOR_10D['frac_over_0.5std']) < 1e-12)
    anchor['anchor_10d_recomputed'] = a10
    log(f'[A] anchor 10D: {a10}')
    assert a10['match'], '10D anchor mismatch — 协议复现失败，停止'

    # ---- 主分析：25D，3 档阈值；10D 3 档作对照 ----
    sens_rows = []
    pairs_by = {}
    for space, cols in [('10D_original', list(e19.MATCH_COLS)),
                        ('25D_full_M0', m0_cols)]:
        for fr in THR_FRACS:
            df = build_pairs_with_cols(val, cols, thr_frac=fr)
            pairs_by[(space, fr)] = df
            m = pair_metrics(df)
            row = {'matching_space': space, 'n_match_cols': len(cols),
                   'thr_frac': fr, 'n_pairs': m['n_pairs'],
                   'n_scenarios': m['n_scenarios'],
                   'pair_dist_median': m['pair_dist_median'],
                   'median_dJ_over_std': m['median_dJ_over_std']['point'],
                   'median_ci_lo': m['median_dJ_over_std']['ci_lo'],
                   'median_ci_hi': m['median_dJ_over_std']['ci_hi']}
            for c in STD_CUTS:
                b = m[f'frac_over_{c}std']
                row[f'frac_over_{c}std'] = b['point']
                row[f'frac_{c}_ci_lo'] = b['ci_lo']
                row[f'frac_{c}_ci_hi'] = b['ci_hi']
            sens_rows.append(row)
            log(f'[A] {space} thr×{fr}: pairs={m["n_pairs"]} '
                f'median={row["median_dJ_over_std"]:.4f} '
                f'>0.5σ={row["frac_over_0.5std"]:.4f}')
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(OUT_DIR, 'full25_aliasing_sensitivity.csv'),
                index=False)

    # ---- 主表：10D vs 25D（thr_frac=1.0）----
    main_rows = [r for r in sens_rows if r['thr_frac'] == 1.0]
    pd.DataFrame(main_rows).to_csv(
        os.path.join(OUT_DIR, 'full25_aliasing_summary.csv'), index=False)

    # ---- 主配对落盘（25D, thr×1.0）----
    p25 = pairs_by[('25D_full_M0', 1.0)].copy()
    p25.insert(0, 'pair_id', [f'full25_{i}' for i in range(len(p25))])
    p25.to_csv(os.path.join(OUT_DIR, 'full25_aliasing_pairs.csv'), index=False)

    # ---- 描述性：pair distance vs |ΔJ*|（仅描述，不作主证据）----
    from scipy.stats import spearmanr
    desc = {}
    for space in ('10D_original', '25D_full_M0'):
        df = pairs_by[(space, 1.0)]
        rho, p = spearmanr(df['pair_dist'], df['dJ_over_std'])
        desc[space] = {'spearman_pairdist_vs_dJnorm': float(rho),
                       'p_value': float(p), 'n_pairs': int(len(df))}
    anchor['descriptive_distance_vs_dJ'] = desc
    anchor['note_effective_dim'] = (
        'M0 25 列中 k_prefix+10 场景常数在同 (uid,k) 组内恒定，标准化后零贡献；'
        '有效匹配维=15 个即时量（10D 原口径为其子集，新增 tgt_dx/dy/dz/'
        'horiz_dist 4 个方向分解 + remaining 已含）。')
    anchor['seed'] = SEED
    anchor['n_boot'] = N_BOOT
    with open(os.path.join(OUT_DIR, 'full25_aliasing_anchor.json'), 'w',
              encoding='utf-8') as f:
        json.dump(anchor, f, ensure_ascii=False, indent=2)

    # ---- 图 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    d10 = pairs_by[('10D_original', 1.0)]['dJ_over_std'].to_numpy()
    d25 = pairs_by[('25D_full_M0', 1.0)]['dJ_over_std'].to_numpy()
    ax = axes[0]
    for v, lab, c in [(d10, '10D original (n=6,593)', '#888888'),
                      (d25, f'25D full M0 (n={len(d25):,})', '#2166ac')]:
        xs = np.sort(v)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label=lab, color=c,
                lw=1.8)
    for c0 in STD_CUTS:
        ax.axvline(c0, color='k', ls=':', lw=0.8, alpha=0.5)
    ax.set_xscale('symlog', linthresh=0.01)
    ax.set_xlabel('|ΔJ*| / σ_scenario')
    ax.set_ylabel('ECDF')
    ax.set_title('(a) Aliasing heavy tail: 10D vs 25D matching')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    x = np.arange(len(THR_FRACS))
    w = 0.36
    for i, (space, c) in enumerate([('10D_original', '#888888'),
                                    ('25D_full_M0', '#2166ac')]):
        sub = sens[sens['matching_space'] == space].sort_values('thr_frac')
        ax.bar(x + (i - 0.5) * w, sub['frac_over_0.5std'], w,
               yerr=[sub['frac_over_0.5std'] - sub['frac_0.5_ci_lo'],
                     sub['frac_0.5_ci_hi'] - sub['frac_over_0.5std']],
               capsize=3, color=c, alpha=0.85,
               label='10D original' if space == '10D_original' else '25D full M0',
               error_kw=dict(lw=1))
    ax.set_xticks(x, [f'{fr}× median' for fr in THR_FRACS])
    ax.set_ylabel('P(|ΔJ*| > 0.5σ)')
    ax.set_xlabel('distance threshold')
    ax.set_title('(b) Threshold sensitivity')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, axis='y')

    ax = axes[2]
    df = pairs_by[('25D_full_M0', 1.0)]
    ax.scatter(df['pair_dist'], df['dJ_over_std'], s=3, alpha=0.25,
               color='#2166ac', rasterized=True)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('25D pair distance (within-scenario z)')
    ax.set_ylabel('|ΔJ*| / σ_scenario')
    rho = desc['25D_full_M0']['spearman_pairdist_vs_dJnorm']
    ax.set_title(f'(c) Distance vs |ΔJ*| (descriptive, Spearman ρ={rho:.3f})')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'figure_full25_aliasing.png'), dpi=600)
    fig.savefig(os.path.join(OUT_DIR, 'figure_full25_aliasing.pdf'))
    log(f'[A done] -> {OUT_DIR}')


if __name__ == '__main__':
    main()
