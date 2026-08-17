# -*- coding: utf-8 -*-
"""
exp31_targeted_specificity.py — Targeted Correction Specificity Audit
（v15 论文补强实验；POST-SELECTION 性质：T 已由 exp21 冻结的
performance–complexity 规则选定，本实验只审计 G/R/T/S 中哪类工程信息
最针对已诊断的 checkpoint-state aliasing，以及 M0+T 相对其他纯增强臂的
ranking/burden-specific 表现。不修改任何冻结对象。）

=====================================================================
审计锚点（2026-08-17 主代理核实；脚本内含断言，失败即停止）
=====================================================================
- 特征组：config/structured_history_groups_v1.json
  G_geometry_evolution=26, R_design_resource=5, T_target_approach=10,
  S_safety_corridor=4（exp8.load_groups 断言并集=C−B 45 列无重叠）。
- matched pairs：paper_ftg_package/revision_20260730/
  aliased_pair_separation.csv（exp29 落盘，6,593 对；配对管线 =
  exp19.build_within_pairs(val, pool_across_k=False, thr_frac=1.0)；
  锚点 median dJ_over_std≈0.013）。本实验不重匹配。
- 距离口径（exp29）：场景内 z-standardization（ddof=0，std=0→1）后
  Euclidean；strata：close = d_match ≤ median(d_match)；
  high-aliasing = close 且 dJ_over_std>0.5；
  low-aliasing = close 且 dJ_over_std<0.25。
- aliasing burden（exp24）：matched_pair_audit.csv 中
  scope∈{within_scenario_samek, within_scenario_anyk} 且
  boundary_consistent=True 的同场景对里 dJ_over_std>0.5 的比例；
  分层 = val burden 四分位 low=Q1 / mid=Q2–Q3 / high=Q4。
- HGBR：e2.train_ensemble（5 成员 HistGradientBoostingRegressor，
  max_iter=300，seed=2024，成员0=全量、成员1–4=scenario cluster
  bootstrap，其余库默认）。新臂 BG/BR/BS 用同一函数同一口径训练。
- 逐场景 ranking 指标：exp24 val_side 协议（pairwise acc：
  e2.build_cell_table cell=(uid,k)；top-1 regret：
  e3.cell_selection_table cell=(uid,group,k)，score=ensemble mean，
  β=0）。
- 冻结模型复用：models/exp21/model_m0.joblib（M0）、model_m1（BT）、
  model_m4（BFULL）；新训练仅 BG=M0+G(51D)、BR=M0+R(30D)、
  BS=M0+S(29D) → models/exp31/。

输出：results/exp_targeted_correction_specificity/（新建）

运行：
  python exp31_targeted_specificity.py --stage audit
  python exp31_targeted_specificity.py --stage mechanism
  python exp31_targeted_specificity.py --stage train      # CPU 重，避开计时实验
  python exp31_targeted_specificity.py --stage ranking
  python exp31_targeted_specificity.py --stage burden
  python exp31_targeted_specificity.py --stage figures
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exp2_train_abc as e2  # noqa: E402
import exp3_conservative_ranking as e3  # noqa: E402
import exp21_feature_ladder as e21  # noqa: E402

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
SEED = 2024
N_BOOT = 1000
HI_CUT, LO_CUT = 0.5, 0.25       # exp29 冻结分层阈值（×场景内标签 std）
BURDEN_CUT = 0.5                 # exp24 冻结 burden 阈值
WITHIN_SCOPES = ('within_scenario_samek', 'within_scenario_anyk')

REV_DIR = os.path.join(_REPO_ROOT, 'paper_ftg_package', 'revision_20260730')
PAIRS_CSV = os.path.join(REV_DIR, 'aliased_pair_separation.csv')   # exp29 落盘
MPA_CSV = os.path.join(REV_DIR, 'matched_pair_audit.csv')          # exp19 落盘
VAL_CSV = os.path.join(_PKG_DIR, 'data', 'val_hr_labels.csv')
TRAIN_CSV = os.path.join(_PKG_DIR, 'data', 'train_hr_labels.csv')
EXP24_CSV = os.path.join(_PKG_DIR, 'results',
                         'exp24_aliasing_burden_analysis.csv')
CONFIG_JSON = os.path.join(_PKG_DIR, 'config',
                           'structured_history_groups_v1.json')
EXP19_PATH = os.path.join(_SCRIPT_DIR, 'exp19_aliasing_audit.py')
EXP29_PATH = os.path.join(_SCRIPT_DIR, 'exp29_pair_separation.py')

MODELS21 = os.path.join(_PKG_DIR, 'models', 'exp21')
MODELS31 = os.path.join(_PKG_DIR, 'models', 'exp31')
OUT_DIR = os.path.join(_PKG_DIR, 'results',
                       'exp_targeted_correction_specificity')

GROUP_KEYS = {'G': 'G_geometry_evolution', 'R': 'R_design_resource',
              'T': 'T_target_approach', 'S': 'S_safety_corridor'}
EXPECT_DIM = {'G': 26, 'R': 5, 'T': 10, 'S': 4}

# exp29 硬编码 T 列（脚本内断言与 config 一致）
EXP29_T_COLS = ['bearing_rel_sin', 'bearing_rel_cos',
                'off_target_angle_deg', 't1_dx', 't1_dy', 't1_dz',
                't1_dist', 'off_target1_angle_deg', 'prefix_t1_min_ell',
                'prefix_t1_passed']

# 6 个 ranking 臂：BT/BFULL 复用 exp21 冻结模型
ARM_MODELS = {'M0': os.path.join(MODELS21, 'model_m0.joblib'),
              'BG': os.path.join(MODELS31, 'model_bg.joblib'),
              'BR': os.path.join(MODELS31, 'model_br.joblib'),
              'BT': os.path.join(MODELS21, 'model_m1.joblib'),
              'BS': os.path.join(MODELS31, 'model_bs.joblib'),
              'BFULL': os.path.join(MODELS21, 'model_m4.joblib')}
ARM_DIMS = {'M0': 25, 'BG': 51, 'BR': 30, 'BT': 35, 'BS': 29,
            'BFULL': 70}

# v15 配色（与 figstyle_v15 一致）
C_T = '#0F4D92'
C_G = '#6BAED6'
C_R = '#969696'
C_S = '#C9A227'
C_FULL = '#333333'
C_DARK = '#333333'
C_M0 = '#767676'
C_CTRL = '#D98E32'

ANCHOR_N_PAIRS = 6593
ANCHOR_MEDIAN_DJ = 0.013


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'),
              flush=True)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_groups():
    with open(CONFIG_JSON, encoding='utf-8') as f:
        cfg = json.load(f)
    groups = {short: list(cfg['groups'][key]['features'])
              for short, key in GROUP_KEYS.items()}
    for s, n in EXPECT_DIM.items():
        assert len(groups[s]) == n, f'{s} 组维度 {len(groups[s])} != {n}'
    all45 = [c for s in ('G', 'R', 'T', 'S') for c in groups[s]]
    assert len(set(all45)) == 45, 'G/R/T/S 有重叠列'
    assert set(EXP29_T_COLS) == set(groups['T']), \
        'exp29 硬编码 T 列与 config T 组不一致'
    return groups


def load_val_exp29():
    """与 exp29.main 逐字相同的 val 加载（a_row/b_row 的索引基准）。"""
    val = pd.read_csv(VAL_CSV)
    val = val[val['hr_rebuild_ok'] == 1].reset_index(drop=True)
    val = val[np.isfinite(val[e2.LABEL])].reset_index(drop=True)
    return val


# ---------------------------------------------------------------------------
# 通用 bootstrap helpers（统计单位 = scenario）
# ---------------------------------------------------------------------------
def boot_ci_mean(vals, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float(np.mean(v)) if len(v) else float('nan'), \
            float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.mean(v)), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def paired_boot_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """逐场景配对差 mean(a-b) 的 cluster bootstrap CI。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 2:
        return float('nan'), float('nan'), float('nan'), int(len(d))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    mm = d[idx].mean(axis=1)
    return float(np.mean(d)), float(np.percentile(mm, 2.5)), \
        float(np.percentile(mm, 97.5)), int(len(d))


def boot_median(vals, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    mm = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.percentile(mm, 2.5)), \
        float(np.percentile(mm, 97.5))


# ---------------------------------------------------------------------------
# Stage: audit —— 打印冻结事实 + 锚点校验
# ---------------------------------------------------------------------------
def stage_audit(out_dir):
    groups = load_groups()
    e19 = _load_module(EXP19_PATH, 'exp19_mod')

    val = load_val_exp29()
    log(f'[audit] val rows={len(val)} scenarios='
        f'{val["scenario_uid"].nunique()}')

    # 锚点 1：exp19 配对管线复现 6,593 对
    pairs = e19.build_within_pairs(val, pool_across_k=False, thr_frac=1.0)
    med = float(pairs['dJ_over_std'].median())
    assert len(pairs) == ANCHOR_N_PAIRS, \
        f'配对数 {len(pairs)} != 锚点 {ANCHOR_N_PAIRS}'
    assert abs(med - ANCHOR_MEDIAN_DJ) < 0.005, \
        f'median dJ_over_std={med:.4f} 偏离锚点 {ANCHOR_MEDIAN_DJ}'
    log(f'[audit] 锚点 PASS: n_pairs={len(pairs)}, '
        f'median dJ/std={med:.4f}')

    # 锚点 2：与 exp29 落盘 pairs 一致
    saved = pd.read_csv(PAIRS_CSV)
    mg = pairs.merge(saved, on=['scenario_uid_a', 'k_prefix_a',
                                'a_row', 'b_row'])
    assert len(mg) == ANCHOR_N_PAIRS, \
        f'与 exp29 落盘 pairs 合并后 {len(mg)} != {ANCHOR_N_PAIRS}'
    log('[audit] 锚点 PASS: exp29 落盘 pairs 完全匹配')

    # G/R/S 列有限性检查（T 已由 exp29 隐式证明有限）
    fin = {}
    for s in ('G', 'R', 'S'):
        X = val[groups[s]].to_numpy(dtype=float)
        fin[s] = int((~np.isfinite(X)).sum())
        log(f'[audit] {s} 组非有限元素数 = {fin[s]}')

    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'experiment': 'exp31_targeted_correction_specificity',
        'nature': 'POST-SELECTION specificity audit（T 已由 exp21 冻结 '
                  'performance-complexity 规则选定；本实验不改选择规则）',
        'groups': {s: {'key': GROUP_KEYS[s], 'dim': EXPECT_DIM[s],
                       'features': groups[s]} for s in ('G', 'R', 'T', 'S')},
        't_cols_match_exp29': True,
        'anchors': {'n_pairs': int(len(pairs)),
                    'median_dJ_over_std': med,
                    'pairs_match_exp29_csv': True},
        'nonfinite_cells_GR_S': fin,
        'frozen_thresholds': {'hi_cut': HI_CUT, 'lo_cut': LO_CUT,
                              'burden_cut': BURDEN_CUT,
                              'burden_strata': 'val burden 四分位 '
                                               'low=Q1/mid=Q2-Q3/high=Q4 '
                                               '(exp24)'},
        'reused_modules': {
            'pair_pipeline': 'exp19_aliasing_audit.build_within_pairs',
            'distance_stats': 'exp29_pair_separation helpers',
            'train': 'exp2_train_abc.train_ensemble '
                     '(HGBR 5-member, max_iter=300, seed=2024)',
            'eval': 'exp21/exp24 val 协议 (build_cell_table + '
                    'cell_selection_table, β=0)',
            'frozen_models': ['models/exp21/model_m0.joblib',
                              'models/exp21/model_m1.joblib',
                              'models/exp21/model_m4.joblib']},
        'new_models': ['models/exp31/model_bg.joblib (M0+G, 51D)',
                       'models/exp31/model_br.joblib (M0+R, 30D)',
                       'models/exp31/model_bs.joblib (M0+S, 29D)'],
        'no_changes_to': ['M1 选择规则', 'train/val split', 'J* 标签',
                          'matched pairs', 'HGBR 超参', '已有实验结果'],
    }
    with open(os.path.join(out_dir, 'audit_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'group_feature_lists.json'), 'w',
              encoding='utf-8') as f:
        json.dump({s: groups[s] for s in ('G', 'R', 'T', 'S')},
                  f, ensure_ascii=False, indent=2)
    log('[audit] audit_manifest.json / group_feature_lists.json 落盘')


# ---------------------------------------------------------------------------
# Stage: mechanism —— Experiment A（G/R/T/S 机制特异性）
# ---------------------------------------------------------------------------
def _std_cols_nan(X):
    """exp29._std_cols 的 NaN 容忍版：场景内 ddof=0，std=0→1。"""
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    return (X - mean) / std


def build_group_matrices(val, cols):
    """uid -> (row_index_array, Z)。与 exp29.build_t_matrices 同口径。"""
    out = {}
    for uid, g in val.groupby('scenario_uid'):
        idx = g.index.to_numpy()
        X = g[cols].to_numpy(dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        out[uid] = (idx, _std_cols_nan(X))
    return out


def pair_distances_group(pairs, mats):
    """逐对 Euclidean 距离；任一端含 NaN 则该对距离为 NaN。"""
    pos = {uid: {r: p for p, r in enumerate(idx)}
           for uid, (idx, _z) in mats.items()}
    d = np.full(len(pairs), np.nan)
    for n, row in enumerate(pairs.itertuples(index=False)):
        uid = row.scenario_uid_a
        _idx, Z = mats[uid]
        ia, ib = pos[uid][row.a_row], pos[uid][row.b_row]
        diff = Z[ia] - Z[ib]
        if np.all(np.isfinite(diff)):
            d[n] = np.linalg.norm(diff)
    return d


def _contrast_on_resample(vals, hi_idx_by, lo_idx_by, samp):
    h = np.concatenate([vals[hi_idx_by[u]] for u in samp
                        if len(hi_idx_by[u])]) \
        if any(len(hi_idx_by[u]) for u in samp) else np.array([])
    l = np.concatenate([vals[lo_idx_by[u]] for u in samp
                        if len(lo_idx_by[u])]) \
        if any(len(lo_idx_by[u]) for u in samp) else np.array([])
    if len(h) < 5 or len(l) < 5:
        return float('nan')
    return float(np.median(h) - np.median(l))


def paired_boot_contrast_diff(pairs, col_t, col_x, hi, lo,
                              n_boot=N_BOOT, seed=SEED):
    """同批场景重抽下 (contrast_T - contrast_X) 的 cluster bootstrap。

    共同支撑：只用两列都有限的对。contrast = median(d|high) -
    median(d|low)（pooled pairs，exp29 口径）。"""
    uid = pairs['scenario_uid_a'].to_numpy()
    uniq = np.unique(uid)
    both = (np.isfinite(pairs[col_t].to_numpy(float))
            & np.isfinite(pairs[col_x].to_numpy(float)))
    vt = pairs[col_t].to_numpy(float)
    vx = pairs[col_x].to_numpy(float)
    hi_idx_by = {u: np.where(hi & both & (uid == u))[0] for u in uniq}
    lo_idx_by = {u: np.where(lo & both & (uid == u))[0] for u in uniq}
    obs = (_contrast_on_resample(vt, hi_idx_by, lo_idx_by, uniq)
           - _contrast_on_resample(vx, hi_idx_by, lo_idx_by, uniq))
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        diffs.append(_contrast_on_resample(vt, hi_idx_by, lo_idx_by, samp)
                     - _contrast_on_resample(vx, hi_idx_by, lo_idx_by,
                                             samp))
    diffs = np.asarray(diffs)
    diffs = diffs[np.isfinite(diffs)]
    return float(obs), float(np.percentile(diffs, 2.5)), \
        float(np.percentile(diffs, 97.5))


def stage_mechanism(out_dir):
    e29 = _load_module(EXP29_PATH, 'exp29_mod')
    groups = load_groups()
    pairs = pd.read_csv(PAIRS_CSV)
    assert len(pairs) == ANCHOR_N_PAIRS
    val = load_val_exp29()

    # --- 逐组距离（d_G/d_R/d_S 新增；d_T 重算校验 exp29 落盘） -----------
    t0 = time.time()
    for s in ('T', 'G', 'R', 'S'):
        mats = build_group_matrices(val, groups[s])
        d = pair_distances_group(pairs, mats)
        if s == 'T':
            assert np.allclose(d, pairs['d_T'].to_numpy(float),
                               rtol=1e-9, atol=1e-12), \
                '重算 d_T 与 exp29 落盘不一致（管线漂移）'
            log('[mechanism] d_T 重算与 exp29 落盘逐字一致 ✓')
        else:
            pairs[f'd_{s}'] = d
            log(f'[mechanism] d_{s} 完成, NaN 对数='
                f'{int(np.isnan(d).sum())}')
    log(f'[mechanism] 距离计算用时 {time.time() - t0:.0f}s')

    hi = (pairs['stratum'] == 'high_aliasing').to_numpy()
    lo = (pairs['stratum'] == 'low_aliasing').to_numpy()
    codes = pd.Categorical(pairs['scenario_uid_a']).codes
    absdj = pairs['dJ_over_std'].to_numpy(float)

    # --- 指标 1：high-vs-low separation（exp29 同口径 boot_diff_median） --
    rows = []
    rho_by_group = {}
    for s in ('G', 'R', 'T', 'S'):
        col = f'd_{s}'
        v = pairs[col].to_numpy(float)
        fin = np.isfinite(v)
        vh, vl = v[hi & fin], v[lo & fin]
        pt, cil, cih = e29.boot_diff_median(
            vh, vl, codes[hi & fin], codes[lo & fin])
        med_h, med_l = float(np.median(vh)), float(np.median(vl))
        # --- 指标 2：场景内 Spearman(d, dJ_over_std) → 跨场景中位数 ------
        uniq, rho = e29.spearman_by_scenario(v[fin], absdj[fin],
                                             codes[fin])
        r_pt, r_lo, r_hi = e29.boot_median_of_scenario_stat(rho)
        frac_pos = float(np.mean(rho[np.isfinite(rho)] > 0))
        rho_by_group[s] = (uniq, rho)
        rows.append({'group': s, 'dimension': EXPECT_DIM[s],
                     'median_distance_low': med_l,
                     'median_distance_high': med_h,
                     'high_low_ratio': med_h / med_l if med_l else np.nan,
                     'high_low_contrast': float(med_h - med_l),
                     'contrast_CI_low': cil, 'contrast_CI_high': cih,
                     'boot_point': pt,
                     'median_spearman': r_pt,
                     'spearman_CI_low': r_lo, 'spearman_CI_high': r_hi,
                     'fraction_positive_rho': frac_pos,
                     'n_nan_pairs': int((~fin).sum())})
        log(f'[mechanism] {s}: contrast={med_h - med_l:+.3f} '
            f'CI[{cil:+.3f},{cih:+.3f}] | median rho={r_pt:+.3f} '
            f'CI[{r_lo:+.3f},{r_hi:+.3f}] frac_pos={frac_pos:.3f}')
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'mechanism_group_summary.csv'), index=False)

    # --- paired contrasts：T vs G/R/S（同批场景重抽） ---------------------
    pr = []
    for s in ('G', 'R', 'S'):
        obs, cil, cih = paired_boot_contrast_diff(pairs, 'd_T', f'd_{s}',
                                                  hi, lo)
        # paired Spearman 中位数差
        uniq_t, rho_t = rho_by_group['T']
        uniq_s, rho_s = rho_by_group[s]
        assert np.array_equal(uniq_t, uniq_s)
        both = np.isfinite(rho_t) & np.isfinite(rho_s)
        d_rho = rho_t[both] - rho_s[both]
        m_pt, m_lo, m_hi = boot_median(d_rho)
        pr.append({'comparison': f'T-{s}',
                   'delta_contrast': obs,
                   'contrast_CI_low': cil, 'contrast_CI_high': cih,
                   'delta_median_rho': m_pt,
                   'rho_CI_low': m_lo, 'rho_CI_high': m_hi,
                   'n_scenarios_rho': int(both.sum())})
        log(f'[mechanism] paired T-{s}: Δcontrast={obs:+.3f} '
            f'CI[{cil:+.3f},{cih:+.3f}] | Δrho={m_pt:+.3f} '
            f'CI[{m_lo:+.3f},{m_hi:+.3f}]')
    pd.DataFrame(pr).to_csv(
        os.path.join(out_dir, 'mechanism_paired_contrasts.csv'),
        index=False)

    pairs.to_csv(os.path.join(out_dir, 'mechanism_pairs_raw.csv'),
                 index=False)
    log('[mechanism] mechanism_group_summary/paired_contrasts/'
        'pairs_raw 落盘')


# ---------------------------------------------------------------------------
# Stage: train —— BG/BR/BS 三个 pure augmentation 臂（冻结 HGBR 口径）
# ---------------------------------------------------------------------------
def stage_train(out_dir):
    os.makedirs(MODELS31, exist_ok=True)
    feat_map, groups, _hist45 = e21.build_feature_map()
    base = set(feat_map['M0'])
    keep = lambda extra: [c for c in e2.FEATURES_C  # noqa: E731
                          if c in base | set(extra)]
    new_arms = {'BG': keep(groups['G']), 'BR': keep(groups['R']),
                'BS': keep(groups['S'])}
    for a, feats in new_arms.items():
        assert len(feats) == ARM_DIMS[a], \
            f'{a} 维度 {len(feats)} != {ARM_DIMS[a]}'

    df_tr, _df_va = e21.load_data(TRAIN_CSV, VAL_CSV)
    log(f'[train] train rows={len(df_tr)} scenarios='
        f'{df_tr["scenario_uid"].nunique()}')
    timing = {}
    for name, feats in new_arms.items():
        mp = ARM_MODELS[name]
        if os.path.exists(mp):
            log(f'[train] {name} 已存在，跳过（断点续跑）')
            continue
        log(f'[train] {name}: {len(feats)}D, 5-member HGBR ...')
        t0 = time.time()
        members = e2.train_ensemble(df_tr, feats)
        wall = time.time() - t0
        joblib.dump({'arm': name, 'features': feats, 'members': members,
                     'train_sec': wall, 'seed': e2.SEED,
                     'max_iter': e2.MAX_ITER,
                     'ensemble_k': e2.ENSEMBLE_K,
                     'note': 'exp31 pure augmentation arm；训练口径与 '
                             'exp21 逐字相同（e2.train_ensemble 默认参数）'},
                    mp)
        timing[name] = {'n_features': len(feats), 'train_sec': wall}
        log(f'[train] {name} done in {wall / 60:.1f} min -> {mp}')
    tp = os.path.join(out_dir, 'train_timing.json')
    if os.path.exists(tp):
        with open(tp, encoding='utf-8') as f:
            old = json.load(f)
        old.update(timing)
        timing = old
    with open(tp, 'w', encoding='utf-8') as f:
        json.dump(timing, f, indent=2)
    log('[train] train_timing.json 落盘')


# ---------------------------------------------------------------------------
# 共用评估：6 臂在 val 上的全局 + 逐场景 ranking 指标（缓存）
# ---------------------------------------------------------------------------
def evaluate_arms(out_dir):
    cache = os.path.join(out_dir, 'per_scenario_metrics.csv')
    gcache = os.path.join(out_dir, 'global_metrics.json')
    if os.path.exists(cache) and os.path.exists(gcache):
        return pd.read_csv(cache), json.load(open(gcache))

    _df_tr, df_va = e21.load_data(TRAIN_CSV, VAL_CSV)
    arms = list(ARM_MODELS)
    preds = {}
    for arm in arms:
        payload = joblib.load(ARM_MODELS[arm])
        assert len(payload['features']) == ARM_DIMS[arm]
        X = df_va[payload['features']].to_numpy(dtype=float)
        preds[arm] = e2.predict_ensemble_mean(payload['members'], X)
        log(f'[eval] {arm} 预测完成')

    uids = np.array(sorted(df_va['scenario_uid'].unique()))

    # 全局指标（exp21 协议：pooled + scenario cluster bootstrap）
    df_ev2 = df_va[['scenario_uid', e2.K_COL, e2.LABEL]].copy()
    for m in arms:
        df_ev2[f'pred_{m}'] = preds[m]
    cells2 = e2.build_cell_table(df_ev2, {m: f'pred_{m}' for m in arms})
    obs_rank = e2._agg_metrics(cells2, arms)
    rank_ci = e21.bootstrap_rank_metrics(cells2, arms, uids,
                                         n_boot=N_BOOT, seed=SEED)

    df_ev3 = df_va[['scenario_uid', 'group_id', e2.K_COL, 'branch_id',
                    e2.LABEL]].copy()
    sel_ci = {}
    sel_cells = {}
    for m in arms:
        ct = e3.cell_selection_table(df_ev3.assign(__s=preds[m]), '__s')
        sel_cells[m] = ct
        sel_ci[m] = e3.metrics_with_ci(ct, uids, n_boot=N_BOOT,
                                       seed=SEED)
        log(f'[eval] {m} selection cells={len(ct)}')

    glob = {}
    for m in arms:
        glob[m] = {
            'dimension': ARM_DIMS[m],
            'pairwise_accuracy':
                obs_rank[m]['pairwise_accuracy'],
            'pairwise_accuracy_ci': [
                rank_ci[m]['pairwise_accuracy']['ci95'][0],
                rank_ci[m]['pairwise_accuracy']['ci95'][1]],
            'top1_regret_mean': sel_ci[m]['top1_regret_mean'],
            'top1_regret_mean_ci': [sel_ci[m]['top1_regret_mean__ci_lo'],
                                    sel_ci[m]['top1_regret_mean__ci_hi']],
            'top1_regret_median': sel_ci[m]['top1_regret_median'],
            'top3_recall': sel_ci[m]['top3_recall'],
        }

    # 逐场景指标（exp24 val_side 协议）
    acc = {}
    for uid, g in cells2.groupby('scenario_uid'):
        acc[uid] = {a: (g[f'{a}__n_correct'].sum()
                        / g[f'{a}__n_pairs'].sum()
                        if g[f'{a}__n_pairs'].sum() > 0 else np.nan)
                    for a in arms}
    per_rows = []
    for m in arms:
        sct = sel_cells[m].groupby('scenario_uid').agg(
            top1_regret=('top1_regret', 'mean'),
            top3_recall=('top3_recall', 'mean'))
        for uid in uids:
            per_rows.append({'scenario_uid': uid, 'arm': m,
                             'dimension': ARM_DIMS[m],
                             'pairwise_acc': acc[uid][m],
                             'top1_regret':
                                 float(sct.loc[uid, 'top1_regret'])
                                 if uid in sct.index else np.nan,
                             'top3_recall':
                                 float(sct.loc[uid, 'top3_recall'])
                                 if uid in sct.index else np.nan})
    per = pd.DataFrame(per_rows)
    per.to_csv(cache, index=False)
    with open(gcache, 'w', encoding='utf-8') as f:
        json.dump(glob, f, indent=2)
    log('[eval] per_scenario_metrics.csv / global_metrics.json 落盘')
    return per, glob


# ---------------------------------------------------------------------------
# Stage: ranking —— Experiment B（pure augmentation arms 对比）
# ---------------------------------------------------------------------------
def stage_ranking(out_dir):
    per, glob = evaluate_arms(out_dir)

    rows = []
    pv = per.pivot(index='scenario_uid', columns='arm',
                   values='top1_regret')
    pa = per.pivot(index='scenario_uid', columns='arm',
                   values='pairwise_acc')
    for m in ARM_MODELS:
        g = glob[m]
        d_reg, d_rlo, d_rhi, n1 = paired_boot_ci(
            pv['M0'].to_numpy(), pv[m].to_numpy())
        d_acc, d_alo, d_ahi, _n2 = paired_boot_ci(
            pa[m].to_numpy(), pa['M0'].to_numpy())
        rows.append({'arm': m, 'dimension': g['dimension'],
                     'pairwise_accuracy': g['pairwise_accuracy'],
                     'accuracy_CI_low': g['pairwise_accuracy_ci'][0],
                     'accuracy_CI_high': g['pairwise_accuracy_ci'][1],
                     'mean_top1_regret': g['top1_regret_mean'],
                     'regret_CI_low': g['top1_regret_mean_ci'][0],
                     'regret_CI_high': g['top1_regret_mean_ci'][1],
                     'delta_regret_vs_M0': d_reg,
                     'delta_regret_CI': [d_rlo, d_rhi],
                     'delta_accuracy_vs_M0': d_acc,
                     'delta_accuracy_CI': [d_alo, d_ahi]})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'pure_arm_ranking_summary.csv'),
        index=False)

    # paired BT vs alternatives
    pr = []
    for x in ('BG', 'BR', 'BS', 'BFULL'):
        d_reg, rlo, rhi, n = paired_boot_ci(pv[x].to_numpy(),
                                            pv['BT'].to_numpy())
        d_acc, alo, ahi, _ = paired_boot_ci(pa['BT'].to_numpy(),
                                            pa[x].to_numpy())
        pr.append({'comparison': f'BT vs {x}',
                   'delta_regret_X_minus_T': d_reg,
                   'regret_CI_low': rlo, 'regret_CI_high': rhi,
                   'delta_accuracy_T_minus_X': d_acc,
                   'accuracy_CI_low': alo, 'accuracy_CI_high': ahi,
                   'n_scenarios': n})
        log(f'[ranking] BT vs {x}: Δregret={d_reg:+.4g} '
            f'CI[{rlo:+.4g},{rhi:+.4g}] | Δacc={d_acc:+.4f} '
            f'CI[{alo:+.4f},{ahi:+.4f}]')
    pd.DataFrame(pr).to_csv(
        os.path.join(out_dir, 'ranking_paired_BT_vs_alternatives.csv'),
        index=False)

    # performance-complexity 表（Experiment D 数据）
    pc = pd.DataFrame([{'arm': m, 'dimension': glob[m]['dimension'],
                        'pairwise_accuracy': glob[m]['pairwise_accuracy'],
                        'mean_top1_regret': glob[m]['top1_regret_mean']}
                       for m in ARM_MODELS])
    pc.to_csv(os.path.join(out_dir, 'performance_complexity.csv'),
              index=False)
    log('[ranking] pure_arm_ranking_summary / paired / '
        'performance_complexity 落盘')


# ---------------------------------------------------------------------------
# Stage: burden —— Experiment C（problem-severity-specific comparison）
# ---------------------------------------------------------------------------
def _compute_burden():
    mpa = pd.read_csv(MPA_CSV)
    w = mpa[mpa['scope'].isin(WITHIN_SCOPES)
            & (mpa['boundary_consistent'] == True)]  # noqa: E712
    burden = w.groupby('scenario_uid_a').apply(
        lambda g: float((g['dJ_over_std'] > BURDEN_CUT).mean()),
        include_groups=False)
    # 与 exp24 落盘逐字核对
    e24 = pd.read_csv(EXP24_CSV)
    e24v = e24[e24['split'] == 'val'].set_index('scenario_uid')['burden']
    b = burden.rename('burden').reset_index().rename(
        columns={'scenario_uid_a': 'scenario_uid'}).set_index(
        'scenario_uid')['burden']
    common = e24v.index.intersection(b.index)
    # 差异上限 ~5e-9，为浮点除法顺序噪声（如 1/3 的不同求和顺序）；
    # exp24 落盘仅 180 个 val 标签场景，重算含 MPA 中全部 186 个
    # （下游与 per-scenario 指标 join 后自然限制到 180）。
    assert np.allclose(b.loc[common].to_numpy(float),
                       e24v.loc[common].to_numpy(float),
                       rtol=1e-6, atol=1e-6), \
        'burden 重算与 exp24 落盘不一致（超出浮点噪声）'
    log(f'[burden] 与 exp24 落盘 burden 一致（容差 1e-6，'
        f'max|diff|≤5e-9 浮点噪声；n={len(common)} 场景）')
    return b


def stage_burden(out_dir):
    per, _glob = evaluate_arms(out_dir)
    burden = _compute_burden()

    pv = per.pivot(index='scenario_uid', columns='arm',
                   values='top1_regret')
    df = pv.copy()
    df['burden'] = burden
    df = df.dropna(subset=['burden'])
    uids = df.index.to_numpy()

    # exp24 冻结四分位分层（low=Q1 / mid=Q2-Q3 / high=Q4）
    q1, _q2, q3 = df['burden'].quantile([0.25, 0.5, 0.75]).to_numpy()
    strata = pd.cut(df['burden'], [-np.inf, q1, q3, np.inf],
                    labels=['low', 'mid', 'high'])
    df['stratum'] = strata.to_numpy()

    # 每臂 gain = regret_M0 - regret_arm
    gain_cols = {}
    for m in ARM_MODELS:
        if m == 'M0':
            continue
        df[f'gain_{m}'] = df['M0'] - df[m]
        gain_cols[m] = f'gain_{m}'

    # raw 落盘
    raw = df.reset_index()[['scenario_uid', 'burden', 'stratum']
                           + list(gain_cols.values())]
    raw.to_csv(os.path.join(out_dir, 'burden_specificity_raw.csv'),
               index=False)

    # 分层 summary
    rows = []
    for m, gc in gain_cols.items():
        for s in ('low', 'mid', 'high'):
            v = df.loc[df['stratum'] == s, gc].to_numpy(float)
            pt, lo, hi = boot_ci_mean(v)
            rows.append({'arm': m, 'burden_stratum': s,
                         'n_scenarios': int(np.isfinite(v).sum()),
                         'mean_regret_gain_vs_M0': pt,
                         'CI_low': lo, 'CI_high': hi})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, 'burden_specificity_summary.csv'),
        index=False)

    # 连续 burden：Huber RLM（exp24 同口径）
    import statsmodels.api as sm
    srows = []
    for m, gc in gain_cols.items():
        x = df['burden'].to_numpy(float)
        y = df[gc].to_numpy(float)
        fit = sm.RLM(y, sm.add_constant(x),
                     M=sm.robust.norms.HuberT()).fit()
        b_, se = fit.params[1], fit.bse[1]
        srows.append({'arm': m, 'slope': float(b_),
                      'CI_low': float(b_ - 1.96 * se),
                      'CI_high': float(b_ + 1.96 * se),
                      'p_value': float(fit.pvalues[1]),
                      'n_scenarios': int(len(x))})
        log(f'[burden] {m}: slope={b_:+.4g} '
            f'CI[{b_ - 1.96 * se:+.4g},{b_ + 1.96 * se:+.4g}] '
            f'p={fit.pvalues[1]:.3g}')
    pd.DataFrame(srows).to_csv(
        os.path.join(out_dir, 'table_burden_slopes.csv'), index=False)

    # high 层 paired：Gain_T - Gain_X
    hi_df = df[df['stratum'] == 'high']
    pr = []
    for x in ('BG', 'BR', 'BS', 'BFULL'):
        d, lo, hi_, n = paired_boot_ci(
            hi_df['gain_BT'].to_numpy(), hi_df[f'gain_{x}'].to_numpy())
        pr.append({'comparison': f'BT-{x} (high-burden gain)',
                   'delta_gain': d, 'CI_low': lo, 'CI_high': hi_,
                   'n_scenarios': n})
        log(f'[burden] high 层 BT-{x}: Δgain={d:+.4g} '
            f'CI[{lo:+.4g},{hi_:+.4g}] (n={n})')
    pd.DataFrame(pr).to_csv(
        os.path.join(out_dir,
                     'table_T_vs_alternatives_high_burden.csv'),
        index=False)
    with open(os.path.join(out_dir, 'burden_strata_edges.json'), 'w') as f:
        json.dump({'q1': float(q1), 'q3': float(q3),
                   'rule': 'low=Q1 / mid=Q2-Q3 / high=Q4 (exp24 冻结)'},
                  f, indent=2)
    log('[burden] burden_specificity raw/summary + slopes + '
        'high-burden paired 落盘')


# ---------------------------------------------------------------------------
# Stage: figures
# ---------------------------------------------------------------------------
def stage_figures(out_dir):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt

    mech = pd.read_csv(os.path.join(out_dir,
                                    'mechanism_group_summary.csv'))
    pc = pd.read_csv(os.path.join(out_dir, 'performance_complexity.csv'))
    bs = pd.read_csv(os.path.join(out_dir,
                                  'burden_specificity_summary.csv'))

    colors = {'G': C_G, 'R': C_R, 'T': C_T, 'S': C_S}
    arm_colors = {'M0': C_M0, 'BG': C_G, 'BR': C_R, 'BT': C_T,
                  'BS': C_S, 'BFULL': C_FULL}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) high-low contrast
    ax = axes[0]
    xs = np.arange(len(mech))
    ax.errorbar(xs, mech['high_low_contrast'],
                yerr=[mech['high_low_contrast'] - mech['contrast_CI_low'],
                      mech['contrast_CI_high'] - mech['high_low_contrast']],
                fmt='none', capsize=4, ecolor='#444444', elinewidth=1.2)
    for i, g in enumerate(mech['group']):
        ax.plot(xs[i], mech['high_low_contrast'][i], 'o',
                color=colors[g], markersize=9)
    ax.axhline(0, color=C_DARK, lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{g} ({d}D)' for g, d in
                        zip(mech['group'], mech['dimension'])])
    ax.set_ylabel('High - low aliasing\nmedian distance contrast')
    ax.set_title('(a) Problem correspondence', fontsize=10, loc='left')

    # (b) performance-complexity
    ax = axes[1]
    for _i, r in pc.iterrows():
        ax.plot(r['dimension'], r['mean_top1_regret'], 'o',
                color=arm_colors[r['arm']], markersize=9)
        dy = {'BT': -0.9, 'M0': 0.55, 'BFULL': 0.55}.get(r['arm'], 0.55)
        ax.annotate(r['arm'], (r['dimension'], r['mean_top1_regret']),
                    textcoords='offset points', xytext=(6, dy * 10),
                    fontsize=8.5)
    ax.set_xlabel('State dimension')
    ax.set_ylabel('Mean top-1 regret')
    ax.set_title('(b) Performance-complexity', fontsize=10, loc='left')
    ax.set_xlim(15, 80)

    # (c) burden strata × gain
    ax = axes[2]
    xpos = {'low': 0, 'mid': 1, 'high': 2}
    for m in ('BG', 'BR', 'BT', 'BS', 'BFULL'):
        g = bs[bs['arm'] == m].set_index('burden_stratum')
        ys = [g.loc[s, 'mean_regret_gain_vs_M0']
              for s in ('low', 'mid', 'high')]
        lo = [g.loc[s, 'CI_low'] for s in ('low', 'mid', 'high')]
        hi_ = [g.loc[s, 'CI_high'] for s in ('low', 'mid', 'high')]
        xs = [xpos[s] for s in ('low', 'mid', 'high')]
        ax.errorbar(xs, ys,
                    yerr=[np.array(ys) - np.array(lo),
                          np.array(hi_) - np.array(ys)],
                    fmt='-o', color=arm_colors[m], label=m, capsize=3,
                    lw=1.5, markersize=6)
    ax.axhline(0, color=C_DARK, lw=0.8, ls='--')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Low', 'Medium', 'High'])
    ax.set_xlabel('Aliasing-burden stratum')
    ax.set_ylabel('Top-1 regret gain vs M0')
    ax.set_title('(c) Severity-specific benefit', fontsize=10, loc='left')
    ax.set_ylim(-16, 12)   # 裁剪 BG high 层超长 CI 下端，保留形状可读
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fp = os.path.join(out_dir,
                          f'figure_targeted_correction_600dpi.{ext}')
        fig.savefig(fp, dpi=600 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    log('[figures] figure_targeted_correction_600dpi.png/.pdf 落盘')


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['audit', 'mechanism', 'train', 'ranking',
                             'burden', 'figures', 'all'])
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    log(f'[exp31] out={OUT_DIR} stage={args.stage}')
    t_all = time.time()

    if args.stage in ('all', 'audit'):
        stage_audit(OUT_DIR)
    if args.stage in ('all', 'mechanism'):
        stage_mechanism(OUT_DIR)
    if args.stage in ('all', 'train'):
        stage_train(OUT_DIR)
    if args.stage in ('all', 'ranking'):
        stage_ranking(OUT_DIR)
    if args.stage in ('all', 'burden'):
        stage_burden(OUT_DIR)
    if args.stage in ('all', 'figures'):
        stage_figures(OUT_DIR)

    log(f'[exp31] total {(time.time() - t_all) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
