# -*- coding: utf-8 -*-
"""sse_expC_pair_decision.py — Exp C：diagnosed pairs 上的 decision-level 机制检验。

任务书：state_sufficiency_extension / Exp C。
问题：T-space 分离（Figure 5）是否真的转化成 HGBR 把 aliased pairs 的 J*
顺序排对？

pair 总体：主 = Exp A full-25D 配对（full25_aliasing_pairs.csv）；
sensitivity = exp31 的 10D 配对（mechanism_pairs_raw.csv）。

模型（全部冻结/新训 joblib 直接对 val180 预测，确定性、不重训）：
  C0 = M0            models/exp21/model_m0.joblib
  C1 = M0+current8   models/sse/model_b1.joblib   （Exp B）
  C2 = M0+history2   models/sse/model_b2.joblib   （Exp B）
  C3 = M0+fullT      models/exp21/model_m1.joblib
  C4 = shuffle-T     models/exp21/model_m5.joblib （C-Shuffle 对照）
  C5 = random-count  models/exp21/model_m6.joblib （feature-count 对照）

指标：pair ordering accuracy = mean[ sign(ΔĴ)==sign(ΔJ*) ]，分层
all / |ΔJ*|>0.5σ / >1.0σ。near-tie：主口径排除 |ΔJ*|<0.35（noise band）；
sensitivity 口径保留全部（仅排除 |ΔJ*|<1e-9 的精确 tie）。
CI：scenario cluster bootstrap（B=1000, seed=2024），模型间对比用同一
重抽样的配对差。

输出：experiments/state_sufficiency_extension/expC_pair_decision/
  pair_decision_accuracy.csv  pair_decision_summary.csv
  figure_pair_decision.png

运行：python .../sse_expC_pair_decision.py
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import exp2_train_abc as e2  # noqa: E402
import exp21_feature_ladder as e21  # noqa: E402

VAL_CSV = os.path.join(_PKG_DIR, 'data', 'val_hr_labels.csv')
TRAIN_CSV = os.path.join(_PKG_DIR, 'data', 'train_hr_labels.csv')
MODELS21 = os.path.join(_PKG_DIR, 'models', 'exp21')
MODELS_SSE = os.path.join(_PKG_DIR, 'models', 'sse')
EXPA_PAIRS = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension',
                          'expA_full25_aliasing', 'full25_aliasing_pairs.csv')
EXP31_PAIRS = os.path.join(_PKG_DIR, 'results',
                           'exp_targeted_correction_specificity',
                           'mechanism_pairs_raw.csv')
OUT_DIR = os.path.join(_REPO, 'experiments', 'state_sufficiency_extension',
                       'expC_pair_decision')

SEED = 2024
N_BOOT = 1000
NOISE_BAND = 0.35
MODELS = {'C0_M0': os.path.join(MODELS21, 'model_m0.joblib'),
          'C1_current8': os.path.join(MODELS_SSE, 'model_b1.joblib'),
          'C2_history2': os.path.join(MODELS_SSE, 'model_b2.joblib'),
          'C3_fullT': os.path.join(MODELS21, 'model_m1.joblib'),
          'C4_shuffle': os.path.join(MODELS21, 'model_m5.joblib'),
          'C5_random': os.path.join(MODELS21, 'model_m6.joblib')}
STRATA = {'all': 0.0, 'gt_0.5sigma': 0.5, 'gt_1.0sigma': 1.0}


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def load_val():
    """与 exp19/Exp A 逐字一致的过滤（行索引约定）。"""
    val = pd.read_csv(VAL_CSV)
    val = val[val['hr_rebuild_ok'] == 1].reset_index(drop=True)
    val = val[np.isfinite(val['hr_J_star_cont'])].reset_index(drop=True)
    return val


def predict_all(val):
    _tr, va = e21.load_data(TRAIN_CSV, VAL_CSV)
    assert len(va) == len(val), 'e21.load_data 与 exp19 过滤行数不一致'
    preds = {}
    for name, mp in MODELS.items():
        payload = joblib.load(mp)
        X = va[payload['features']].to_numpy(dtype=float)
        preds[name] = e2.predict_ensemble_mean(payload['members'], X)
        log(f'[pred] {name} done')
    return va, preds


def pair_accuracy(pairs, val, preds, tie_rule='noise_band'):
    """逐对 ordering accuracy。tie_rule: 'noise_band'（主，排除 |ΔJ*|<0.35）
    或 'all'（sensitivity，仅排除 |ΔJ*|<1e-9）。"""
    a = pairs['a_row'].to_numpy()
    b = pairs['b_row'].to_numpy()
    dj = (val['hr_J_star_cont'].to_numpy()[a]
          - val['hr_J_star_cont'].to_numpy()[b])
    # 与 pairs 落盘的 abs_dJ 对账
    assert np.allclose(np.abs(dj), pairs['abs_dJ'].to_numpy(),
                       rtol=0, atol=1e-9), 'pair 行索引与 val 不一致'
    res = {}
    for name, p in preds.items():
        dp = p[a] - p[b]
        if tie_rule == 'noise_band':
            keep = np.abs(dj) >= NOISE_BAND
        else:
            keep = np.abs(dj) >= 1e-9
        s_true = np.sign(dj[keep])
        s_pred = np.sign(dp[keep])
        correct = (s_true == s_pred).astype(float)
        res[name] = {'correct': correct,
                     'codes': pd.Categorical(
                         pairs['scenario_uid_a'].to_numpy()[keep]).codes,
                     'dJ_norm': pairs['dJ_over_std'].to_numpy()[keep],
                     'n_pairs': int(keep.sum())}
    return res


def boot_acc(res_entry, cut):
    m = res_entry['dJ_norm'] > cut
    v = res_entry['correct'][m]
    codes = res_entry['codes'][m]
    if len(v) < 2:
        return np.nan, np.nan, np.nan, 0
    uniq = np.unique(codes)
    by = [v[codes == u] for u in uniq]
    rng = np.random.RandomState(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        boots[i] = np.concatenate([by[j] for j in pick]).mean()
    return float(v.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5)), int(len(v))


def boot_acc_paired(re, rf, cut):
    """同一重抽样流下两模型 accuracy 差（re - rf）的 CI。"""
    m = (re['dJ_norm'] > cut) & (rf['dJ_norm'] > cut)
    # 两模型 keep 掩码相同（tie 规则只依赖真值），dJ_norm/codes 一致
    v1, v2 = re['correct'][m], rf['correct'][m]
    codes = re['codes'][m]
    if len(v1) < 2:
        return np.nan, np.nan, np.nan, 0
    uniq = np.unique(codes)
    by1 = [v1[codes == u] for u in uniq]
    by2 = [v2[codes == u] for u in uniq]
    rng = np.random.RandomState(SEED)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        a1 = np.concatenate([by1[j] for j in pick]).mean()
        a2 = np.concatenate([by2[j] for j in pick]).mean()
        boots[i] = a1 - a2
    return float(v1.mean() - v2.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5)), int(len(v1))


def analyze_population(pairs, val, preds, tag):
    rows, paired_rows = [], []
    for tie_rule in ('noise_band', 'all'):
        res = pair_accuracy(pairs, val, preds, tie_rule)
        for sname, cut in STRATA.items():
            for name in MODELS:
                pt, lo, hi, n = boot_acc(res[name], cut)
                rows.append({'population': tag, 'tie_rule': tie_rule,
                             'stratum': sname, 'model': name,
                             'accuracy': pt, 'ci_lo': lo, 'ci_hi': hi,
                             'n_pairs': n})
            if tie_rule == 'noise_band':
                for ref in ('C0_M0', 'C4_shuffle', 'C5_random'):
                    for name in ('C1_current8', 'C2_history2', 'C3_fullT'):
                        if name == ref:
                            continue
                        d, lo, hi, n = boot_acc_paired(res[name], res[ref],
                                                       cut)
                        paired_rows.append({'population': tag,
                                            'stratum': sname,
                                            'comparison': f'{name}-{ref}',
                                            'delta_accuracy': d,
                                            'ci_lo': lo, 'ci_hi': hi,
                                            'n_pairs': n})
    return rows, paired_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    val = load_val()
    _va, preds = predict_all(val)
    assert len(_va) == len(val)

    all_rows, all_paired = [], []
    # 主总体：full-25D pairs
    p25 = pd.read_csv(EXPA_PAIRS)
    r, pr = analyze_population(p25, val, preds, 'full25D')
    all_rows += r
    all_paired += pr
    # sensitivity：10D pairs
    p10 = pd.read_csv(EXP31_PAIRS)
    r, pr = analyze_population(p10, val, preds, 'orig10D')
    all_rows += r
    all_paired += pr

    acc = pd.DataFrame(all_rows)
    acc.to_csv(os.path.join(OUT_DIR, 'pair_decision_accuracy.csv'),
               index=False)
    paired = pd.DataFrame(all_paired)
    paired.to_csv(os.path.join(OUT_DIR, 'pair_decision_summary.csv'),
                  index=False)
    main_tab = acc[(acc['population'] == 'full25D')
                   & (acc['tie_rule'] == 'noise_band')]
    log(f'[main table]\n{main_tab.to_string()}')

    draw_figure(acc)
    log(f'[done] -> {OUT_DIR}')


def draw_figure(acc):
    # ---- 图（主口径 noise_band，full25D）----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    colors = {'C0_M0': '#888888', 'C1_current8': '#4393c3',
              'C2_history2': '#7b3294', 'C3_fullT': '#2166ac',
              'C4_shuffle': '#d95f02', 'C5_random': '#e6ab02'}
    for ax, pop in zip(axes, ['full25D', 'orig10D']):
        sub = acc[(acc['population'] == pop) & (acc['tie_rule']
                                                == 'noise_band')]
        x = np.arange(len(MODELS))
        w = 0.26
        for j, (sname, mark) in enumerate([('all', 'all pairs'),
                                           ('gt_0.5sigma', '>0.5σ'),
                                           ('gt_1.0sigma', '>1.0σ')]):
            s = sub[sub['stratum'] == sname].set_index('model')
            vals = [s.loc[m, 'accuracy'] for m in MODELS]
            lo = [s.loc[m, 'accuracy'] - s.loc[m, 'ci_lo'] for m in MODELS]
            hi = [s.loc[m, 'ci_hi'] - s.loc[m, 'accuracy'] for m in MODELS]
            ax.bar(x + (j - 1) * w, vals, w, yerr=[lo, hi], capsize=2,
                   color=[colors[m] for m in MODELS],
                   alpha=0.35 + 0.3 * j, error_kw=dict(lw=0.8),
                   label=mark if ax is axes[0] else None)
        ax.set_xticks(x, [m.replace('C', 'C') for m in MODELS],
                      fontsize=7, rotation=20)
        ax.set_ylabel('pair ordering accuracy')
        ax.set_title(f'(a) full-25D pairs' if pop == 'full25D'
                     else f'(b) original-10D pairs (sensitivity)')
        ax.grid(alpha=0.25, axis='y')
        ax.set_ylim(0.4, 1.0)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=9,
               loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(OUT_DIR, 'figure_pair_decision.png'), dpi=600)
    fig.savefig(os.path.join(OUT_DIR, 'figure_pair_decision.pdf'))
    log(f'[done] -> {OUT_DIR}')


if __name__ == '__main__':
    if '--figure-only' in sys.argv:
        draw_figure(pd.read_csv(os.path.join(OUT_DIR,
                                             'pair_decision_accuracy.csv')))
    else:
        main()
