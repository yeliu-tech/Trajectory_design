# -*- coding: utf-8 -*-
"""
sse_expF_independent_confirmation.py — Exp F：M0 vs M1 排序效应的
独立 36 场景确认（state_sufficiency_extension / Exp F）。

=====================================================================
背景与纪律（先登记后使用）
=====================================================================
v36 审查 P1a：M0→M1 排序效应此前只在 180 个 validation 场景上验证，
36 个 independent 场景只用于终端等成本测试（结果为 null）。本实验在
**冻结的独立 36 场景决策单元**上直接评价冻结的 M0/M1 模型，不训练、
不重标、不改任何划分——只是 prediction + statistics。

禁止事项（与 v36 审查纪律一致）：
  - 不把 180 验证场景事后拆分冒充独立验证；
  - 36 场景只用于方向确认，无论结果如何都不回调模型/阈值/划分；
  - 负结果如实报告。

=====================================================================
决策单元与数据（逐字复用 exp18 冻结口径）
=====================================================================
  - 单元重建：exp18_sdftg_independent_check.build_units（数据源
    $HCFTG_SCRATCH/exp6_search_checkpoint.csv 五臂 planner 候选；
    标签 exp6_labels_checkpoint.csv，键 (scenario_uid, params_sha256)，
    列 J_star_cont）。单元=(scenario_uid, arm, 候选 sha 集合)，单元内
    ≥2 个有标签候选。
  - 特征：exp18 内容寻址 checkpoint exp18_features_checkpoint.csv
    （69 列 + k_prefix，已核对 M0 25 列 / M1 35 列全覆盖）；
    缺失时按 exp18 同款 worker 链路补算（断点续跑，不动已有行）。
  - 模型：models/exp21/model_m0.joblib（25D）、model_m1.joblib（35D），
    score = 5 成员 ensemble mean μ（与 val 排序协议一致，越小越优）。

=====================================================================
指标与聚合
=====================================================================
主指标（与 val 同口径）：
  - pairwise_accuracy：单元内所有 |ΔJ*|≥1e-9 的候选对，符号一致记 1、
    预测严格相等记 0.5（exp2_train_abc.build_cell_table 同式）；
    pooled = Σn_correct/Σn_pairs。
  - top1_regret：score top-1 的 J* − 单元 min J*（exp18 regret 同定义）。
次指标（与 exp18 连续性）：top3_recall、CFO（阈值 0.70）。
聚合：单元 → scenario（pairwise 按场景 pooled；regret/top3/cfo 场景内
简单平均）→ 跨 36 场景；cluster bootstrap 95% CI（B=1000, seed=2024，
与 exp18 同）。配对 Δ(M1−M0) 按场景配对后 bootstrap。

=====================================================================
预登记判定规则
=====================================================================
  - confirmed：Δpairwise_accuracy = acc_M1 − acc_M0 > 0
    且 Δtop1_regret = regret_M1 − regret_M0 < 0（点估计方向与 val 一致）。
  - 任一项方向不一致 → 如实报告 mixed / not confirmed。
  - CI 只描述不确定性，不用 CI 是否跨零翻转方向判定。
  - 幅度只作描述：独立总体是 planner 决策单元分布，与 val 的均匀
    cell 结构不同，不与 val 做等值检验，只并列展示（口径差异在此登记）。

【输出】experiments/state_sufficiency_extension/expF_independent_confirmation/
  unit_metrics_raw.csv / scenario_metrics.csv / paired_contrast.csv /
  val_comparison.csv / run_manifest.json / technical_report.md（后写）

运行：
  python .../sse_expF_independent_confirmation.py --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import exp18_sdftg_independent_check as e18  # noqa: E402  # 冻结单元/特征链路
from exp2_train_abc import FEATURES_C  # noqa: E402

_PKG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_REPO = os.path.abspath(os.path.join(_PKG_DIR, os.pardir, os.pardir))
OUT_DIR = os.path.join(_REPO, 'experiments',
                       'state_sufficiency_extension',
                       'expF_independent_confirmation')
MODEL_PATHS = {
    'M0': os.path.join(_PKG_DIR, 'models', 'exp21', 'model_m0.joblib'),
    'M1': os.path.join(_PKG_DIR, 'models', 'exp21', 'model_m1.joblib'),
}
EXPB_SUMMARY = os.path.join(_REPO, 'experiments',
                            'state_sufficiency_extension',
                            'expB_T_decomposition', 'summary.csv')

SEED = 2024
N_BOOT = 1000
CAT_THRESHOLD = 0.70          # CFO 阈值（exp18 同）
ARMS = ('M0', 'M1')


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 评分与单元指标
# ---------------------------------------------------------------------------
def load_scorers():
    """返回 {arm: (feature_indices, members)}；score=ensemble mean μ。"""
    scorers = {}
    for arm, path in MODEL_PATHS.items():
        pl = joblib.load(path)
        idx = [FEATURES_C.index(f) for f in pl['features']]
        scorers[arm] = (idx, pl['members'])
        log(f'[model] {arm}: {len(idx)} features, '
            f'{len(pl["members"])} members <- {path}')
    return scorers


def unit_pairwise(y, s):
    """单元内成对统计（exp2_train_abc.build_cell_table 同式）。"""
    n = len(y)
    iu = np.triu_indices(n, k=1)
    d_true = y[iu[0]] - y[iu[1]]
    valid = np.abs(d_true) >= 1e-9
    sgn_true = np.sign(d_true[valid])
    dp = (s[iu[0]] - s[iu[1]])[valid]
    n_pairs = int(valid.sum())
    n_correct = float((np.sign(dp) == sgn_true).sum()
                      + 0.5 * (dp == 0.0).sum())
    return n_pairs, n_correct


def unit_all_metrics(y, s):
    """pairwise + exp18 的 regret/top3/cfo。"""
    n_pairs, n_correct = unit_pairwise(y, s)
    m = e18.unit_metrics(y, s)     # regret / top3_recall / cfo
    m['n_pairs'] = n_pairs
    m['n_correct'] = n_correct
    return m


# ---------------------------------------------------------------------------
# 聚合：scenario 级 + cluster bootstrap
# ---------------------------------------------------------------------------
def scenario_table(unit_rows):
    """unit_rows -> DataFrame(index=scenario_uid)，列：
    (arm, 'acc_np'/'acc_nc'/'regret'/'top3_recall'/'cfo')。
    pairwise 按场景 pooled；其余场景内单元简单平均。"""
    df = pd.DataFrame(unit_rows)
    out = {}
    for uid, g in df.groupby('scenario_uid'):
        for arm in ARMS:
            ga = g[g['arm'] == arm]
            out.setdefault(uid, {})[(arm, 'acc_np')] = ga['n_pairs'].sum()
            out[uid][(arm, 'acc_nc')] = ga['n_correct'].sum()
            for col in ('regret', 'top3_recall', 'cfo'):
                out[uid][(arm, col)] = ga[col].mean()
    sdf = pd.DataFrame(out).T
    sdf.columns = pd.MultiIndex.from_tuples(sdf.columns)
    return sdf


def pooled_acc(sdf, uids, arm, w=None):
    np_ = sdf[(arm, 'acc_np')].reindex(uids).to_numpy(dtype=float)
    nc = sdf[(arm, 'acc_nc')].reindex(uids).to_numpy(dtype=float)
    if w is not None:
        np_, nc = np_ * w, nc * w
    return float(nc.sum() / np_.sum()) if np_.sum() > 0 else np.nan


def scen_metric(sdf, uids, arm, col):
    if col == 'pairwise_accuracy':
        np_ = sdf[(arm, 'acc_np')].reindex(uids).to_numpy(dtype=float)
        nc = sdf[(arm, 'acc_nc')].reindex(uids).to_numpy(dtype=float)
        return np.where(np_ > 0, nc / np.maximum(np_, 1), np.nan)
    return sdf[(arm, col)].reindex(uids).to_numpy(dtype=float)


METRICS = ('pairwise_accuracy', 'regret', 'top3_recall', 'cfo')


def aggregate(sdf, uids, n_boot=N_BOOT, seed=SEED):
    """观测值 + cluster bootstrap CI；配对 Δ(M1−M0)。"""
    rng = np.random.RandomState(seed)
    n_scen = len(uids)
    obs, deltas = {}, {}
    for arm in ARMS:
        obs[arm] = {}
        for met in METRICS:
            if met == 'pairwise_accuracy':
                obs[arm][met] = pooled_acc(sdf, uids, arm)
            else:
                v = scen_metric(sdf, uids, arm, met)
                obs[arm][met] = float(np.nanmean(v))
    for met in METRICS:
        d = scen_metric(sdf, uids, 'M1', met) \
            - scen_metric(sdf, uids, 'M0', met)
        deltas[met] = float(np.nanmean(d))

    boot_obs = {(a, m): [] for a in ARMS for m in METRICS}
    boot_del = {m: [] for m in METRICS}
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n_scen, size=n_scen)
        w = np.bincount(idx, minlength=n_scen).astype(float)
        bu = [uids[i] for i in idx]      # 重复场景按副本计（paired 安全）
        for arm in ARMS:
            boot_obs[(arm, 'pairwise_accuracy')].append(
                pooled_acc(sdf, uids, arm, w=w))
            for met in ('regret', 'top3_recall', 'cfo'):
                v = scen_metric(sdf, uids, arm, met)
                wsum = (w * np.isfinite(v)).sum()
                boot_obs[(arm, met)].append(
                    float((w * np.nan_to_num(v)).sum() / wsum)
                    if wsum > 0 else np.nan)
        for met in METRICS:
            d = scen_metric(sdf, bu, 'M1', met) \
                - scen_metric(sdf, bu, 'M0', met)
            boot_del[met].append(float(np.nanmean(d)))

    def ci(v):
        return [float(np.nanpercentile(v, 2.5)),
                float(np.nanpercentile(v, 97.5))]

    return obs, deltas, {k: ci(v) for k, v in boot_obs.items()}, \
        {m: ci(v) for m, v in boot_del.items()}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(out_dir, workers, n_boot=N_BOOT):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    units = e18.build_units(e18.SEARCH_CKPT, e18.LABEL_CKPT, keep_uids=None)
    uids = np.array(sorted({u['scenario_uid'] for u in units}))
    n_units = len(units)
    log(f'[units] {n_units} units over {len(uids)} scenarios')

    feats = e18.compute_features(units, e18.FEAT_CKPT, workers)
    # 覆盖断言：单元候选必须 100% 有特征
    missing = [(u['scenario_uid'], r['sha']) for u in units
               for r in u['rows']
               if (u['scenario_uid'], r['sha']) not in feats]
    if missing:
        raise RuntimeError(f'特征覆盖缺失 {len(missing)} 条（补算后仍有）')
    log('[feat] coverage 100% OK')

    scorers = load_scorers()

    unit_rows = []
    for u in units:
        uid = u['scenario_uid']
        X70 = np.array([feats[(uid, r['sha'])] for r in u['rows']],
                       dtype=float)
        y = np.array([r['J_star_cont'] for r in u['rows']], dtype=float)
        if not np.isfinite(X70).all():
            log(f'[WARN] 非有限特征，跳过单元 {uid}/{u["arm"]}')
            continue
        for arm in ARMS:
            idx, members = scorers[arm]
            X = X70[:, idx]
            s = np.column_stack([m.predict(X) for m in members]).mean(axis=1)
            met = unit_all_metrics(y, s)
            unit_rows.append({'scenario_uid': uid, 'arm': arm,
                              'search_arm': u['arm'],
                              'n_candidates': len(y), **met})
    log(f'[eval] {len(unit_rows)} unit×arm rows ({time.time() - t0:.0f}s)')

    raw_path = os.path.join(out_dir, 'unit_metrics_raw.csv')
    pd.DataFrame(unit_rows).to_csv(raw_path, index=False, float_format='%.8f')
    log(f'[out] {raw_path}')

    sdf = scenario_table(unit_rows)
    scen_rows = []
    for uid in uids:
        row = {'scenario_uid': uid}
        for arm in ARMS:
            for met in METRICS:
                row[f'{arm}_{met}'] = scen_metric(sdf, [uid], arm, met)[0]
        scen_rows.append(row)
    scen_path = os.path.join(out_dir, 'scenario_metrics.csv')
    pd.DataFrame(scen_rows).to_csv(scen_path, index=False, float_format='%.8f')
    log(f'[out] {scen_path}')

    obs, deltas, obs_ci, del_ci = aggregate(sdf, uids, n_boot=n_boot)

    # ---- 预登记方向判定 ----
    d_acc = deltas['pairwise_accuracy']
    d_reg = deltas['regret']
    verdict = 'confirmed' if (d_acc > 0 and d_reg < 0) else \
        'mixed / not confirmed'
    log(f'[verdict] dAcc={d_acc:+.4f} dRegret={d_reg:+.4f} -> {verdict}')

    # ---- 与 val 并列（expB：B0=M0, B3=M1；口径不同只作描述） ----
    val_rows = []
    if os.path.exists(EXPB_SUMMARY):
        vb = pd.read_csv(EXPB_SUMMARY).set_index('arm')
        for met, col, dcol in (
                ('pairwise_accuracy', 'pairwise_accuracy', None),
                ('regret', 'mean_top1_regret', None)):
            v0, v1 = float(vb.loc['B0', col]), float(vb.loc['B3', col])
            val_rows.append({
                'metric': met,
                'independent_M0': obs['M0'][met],
                'independent_M1': obs['M1'][met],
                'independent_delta': deltas[met],
                'independent_ci_lo': del_ci[met][0],
                'independent_ci_hi': del_ci[met][1],
                'val_M0_B0': v0, 'val_M1_B3': v1,
                'val_delta': v1 - v0,
                'direction_consistent':
                    bool(np.sign(deltas[met]) == np.sign(v1 - v0)),
            })
    vc_path = os.path.join(out_dir, 'val_comparison.csv')
    pd.DataFrame(val_rows).to_csv(vc_path, index=False, float_format='%.8f')
    log(f'[out] {vc_path}')

    pc_rows = []
    for met in METRICS:
        pc_rows.append({
            'metric': met,
            'M0': obs['M0'][met], 'M1': obs['M1'][met],
            'delta_M1_minus_M0': deltas[met],
            'ci_lo': del_ci[met][0], 'ci_hi': del_ci[met][1],
            'M0_ci_lo': obs_ci[('M0', met)][0],
            'M0_ci_hi': obs_ci[('M0', met)][1],
            'M1_ci_lo': obs_ci[('M1', met)][0],
            'M1_ci_hi': obs_ci[('M1', met)][1],
        })
    pc_path = os.path.join(out_dir, 'paired_contrast.csv')
    pd.DataFrame(pc_rows).to_csv(pc_path, index=False, float_format='%.8f')
    log(f'[out] {pc_path}')

    # ---- 分层描述（simple/medium/strong，仅描述不做判定） ----
    try:
        strat = dict(e18.e6.scenario_table())
        dfu = pd.DataFrame(unit_rows)
        dfu['stratum'] = dfu['scenario_uid'].map(strat)
        strat_rows = (dfu.groupby(['stratum', 'arm'])[
            ['regret', 'top3_recall', 'cfo']].mean().reset_index()
            .to_dict(orient='records'))
    except Exception as exc:   # 分层缺失不阻塞主结果
        strat_rows = []
        log(f'[warn] stratum 描述失败：{exc}')

    manifest = {
        'experiment': 'Exp F independent ranking confirmation (M0 vs M1)',
        'preregistered_decision_rule': {
            'confirmed_if': 'delta_pairwise_accuracy > 0 AND '
                            'delta_top1_regret < 0 (point estimates, '
                            'direction match with val)',
            'ci_role': 'descriptive only; no significance-gate flip',
            'magnitude': 'descriptive only; planner decision-unit '
                         'population differs from val uniform cells',
        },
        'verdict': verdict,
        'n_units': int(n_units),
        'n_scenarios': int(len(uids)),
        'n_boot': int(n_boot), 'seed': SEED,
        'inputs': {
            'search_ckpt': {'path': e18.SEARCH_CKPT,
                            'sha256': _sha256(e18.SEARCH_CKPT)},
            'label_ckpt': {'path': e18.LABEL_CKPT,
                           'sha256': _sha256(e18.LABEL_CKPT)},
            'feat_ckpt': {'path': e18.FEAT_CKPT,
                          'sha256': _sha256(e18.FEAT_CKPT)},
            'models': {a: {'path': p, 'sha256': _sha256(p)}
                       for a, p in MODEL_PATHS.items()},
        },
        'obs': {a: {m: obs[a][m] for m in METRICS} for a in ARMS},
        'deltas': deltas, 'delta_ci95': del_ci,
        'stratum_means': strat_rows,
        'elapsed_sec': time.time() - t0,
    }
    man_path = os.path.join(out_dir, 'run_manifest.json')
    with open(man_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f'[out] {man_path}')
    log(f'[done] total {time.time() - t0:.1f}s')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='Exp F：M0/M1 独立 36 场景排序确认')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--n-boot', type=int, default=N_BOOT)
    p.add_argument('--out-dir', type=str, default=OUT_DIR)
    args = p.parse_args(argv)
    for path in (e18.SEARCH_CKPT, e18.LABEL_CKPT, *MODEL_PATHS.values()):
        if not os.path.exists(path):
            print(f'[ERROR] 输入缺失：{path}')
            return 1
    return run(args.out_dir, args.workers, n_boot=args.n_boot)


if __name__ == '__main__':
    sys.exit(main())
