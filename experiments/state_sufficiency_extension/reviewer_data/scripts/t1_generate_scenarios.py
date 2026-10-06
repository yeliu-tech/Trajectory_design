# -*- coding: utf-8 -*-
"""
t1_generate_scenarios.py — 审稿补充数据 T1：60 个全新独立测试场景生成。

=====================================================================
登记（2026-08-29，运行前定稿；此后不改种子、不改 env_id 区间、不改
分层选择规则；结果如何都如实入 manifest）
=====================================================================

【目的】
  为已冻结的 M0/M1 方法提供真正独立的测试场景池。P0 provenance 审计
  已机检确认：rl_retrain 590 场景与 agent 470+120 池精确双射、全部
  进入 988/180 HR 池，无干净历史场景可复用 ⇒ T1 全部新生成。

【生成协议（逐字沿用 exp6 独立场景生成协议，仅换种子流与 env_id 段）】
  - 生成器：neighboring_well_benchmark_v1/rl_retrain/
    generate_training_scenarios.py 的 generate_one_scenario
    （n_seg=9、pop_size=25、n_gen=25、单靶 t1==t2、邻井几何生成器与
    难度抽样不变）。
  - SEED_BASE_T1 = 2500000（与 train 流 910000、agent val 流
    10910000、rl_retrain 流 20250713、exp6 独立流 1500000 均不相交）。
  - 重试规则沿用 generate_all 形式：attempts=1..15，
    base_seed = SEED_BASE_T1 + attempts*131 + env_id；split 取 'val'
    语义（内部再 + env_id*97 + 777777）。
  - env_id 区间：simple 流 7000..，coupled 流 8000..（train ≥1000、
    val ≥2000、rl_retrain 2..600、exp6 独立 5000/6000 段，均不相交）。
  - 分层（每层 20 个，共 60 个；规则同 exp6 仅数量加倍）：
      simple ：neighbor_prob=0.0 流，按 env_id 升序取前 20 个成功者；
      medium ：neighbor_prob=1.0 流中 difficulty=='easy'，升序前 20；
      strong ：neighbor_prob=1.0 流中 difficulty ∈ {moderate,hard}，
               升序前 20；coupled 流上限 120 个 env_id。
  - 落盘 t1_scenarios/：scenario_metadata.json + scenario_XXXX_pareto.csv
    + scenario_XXXX.json（复用 generate_agent_scenarios.
    _meta_to_loader_scenario，经 neighbor_scenario_loader.
    load_scenario(n_seg=9) 逐场景验证）+ manifest.json（生成参数、
    seed、分层标签、逐文件 sha256、与全部已有池的机器核验不相交结果）。

【不相交机器核验】
  1. scenarios_agent/{train,val}：scenario_name 与 random_seed（复用
     exp6._disjointness_check）；
  2. exp6 独立 36 场景 manifest：scenario_uid 与 env_id；
  3. rl_retrain/scenarios/：env_id（文件名 scenario_NNN_pareto.csv）；
  4. 988/180 HR 池 scenario_uid（train_hr_labels/val_hr_labels）。

用法：
  python t1_generate_scenarios.py --workers 32
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RD_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))          # reviewer_data/
_SSE_DIR = os.path.abspath(os.path.join(_RD_DIR, os.pardir))             # state_sufficiency_extension/
_REPO_ROOT = os.path.abspath(os.path.join(_SSE_DIR, os.pardir, os.pardir))
_HCFTG_SCRIPTS = os.path.join(
    _REPO_ROOT, 'experiments', 'history_conditioned_ftg_final', 'scripts')

if _HCFTG_SCRIPTS not in sys.path:
    sys.path.insert(0, _HCFTG_SCRIPTS)

import exp6_independent_validation as e6  # noqa: E402  # 复用常量/助手

# ---------------- T1 登记数值（冻结） ----------------
SEED_BASE_T1 = 2500000
MAX_GEN_ATTEMPTS = 15
N_PER_STRATUM = 20
SIMPLE_START_ID = 7000
COUPLED_START_ID = 8000
COUPLED_MAX_TRIES = 120
N_SEG = 9

T1_DIR = os.path.join(_RD_DIR, 'independent_test', 't1_scenarios')
T1_MANIFEST = os.path.join(T1_DIR, 'manifest.json')


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('gbk', errors='replace').decode('gbk'), flush=True)


def _gen_one(task):
    """worker：单个 env_id 的确定性生成（exp6._gen_one 同款，T1 常量）。"""
    env_id, neighbor_prob = task
    cache_dir = os.path.join(T1_DIR, '_gen_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'env_{env_id}.json')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return env_id, json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    gen = e6._import_from_path('generate_training_scenarios',
                               e6.GENERATOR_PATH)
    result = None
    for attempts in range(1, MAX_GEN_ATTEMPTS + 1):
        result = gen.generate_one_scenario(
            env_id, SEED_BASE_T1 + attempts * 131 + env_id, 'val',
            pop_size=25, n_gen=25, neighbor_prob=neighbor_prob,
            verbose=False)
        if result is not None:
            break
    if result is None:
        return env_id, None
    slim = {'meta': result['meta'], 'records': result['records'],
            'n_attempts': attempts}
    tmp_path = cache_path + f'.tmp{os.getpid()}'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, default=e6._json_default)
    os.replace(tmp_path, cache_path)
    return env_id, slim


def _disjointness_t1(manifest_uids, manifest_names, manifest_seeds,
                     manifest_env_ids):
    """与全部已有池的机器核验不相交。返回 (ok, detail dict)。"""
    out = {}
    # 1. agent train/val（name + seed，复用 exp6 机检）
    ok_agent, out['agent_pools'] = e6._disjointness_check(
        manifest_names, manifest_seeds)
    # 2. exp6 独立 36 场景（uid + env_id）
    with open(e6.INDEP_MANIFEST, 'r', encoding='utf-8') as f:
        m6 = json.load(f)
    u6 = {e['scenario_uid'] for e in m6['scenarios']}
    id6 = {int(e['env_id']) for e in m6['scenarios']}
    out['exp6_independent'] = {
        'n_pool': len(u6),
        'uid_overlap': sorted(manifest_uids & u6),
        'env_id_overlap': sorted(manifest_env_ids & id6),
    }
    # 3. rl_retrain scenarios/（env_id 来自文件名）
    rl_dir = os.path.join(_REPO_ROOT, 'neighboring_well_benchmark_v1',
                          'rl_retrain', 'scenarios')
    rl_ids = set()
    for fn in os.listdir(rl_dir):
        if fn.startswith('scenario_') and fn.endswith('_pareto.csv'):
            try:
                rl_ids.add(int(fn.split('_')[1]))
            except ValueError:
                pass
    out['rl_retrain'] = {
        'n_pool': len(rl_ids),
        'env_id_overlap': sorted(manifest_env_ids & rl_ids),
    }
    # 4. 988/180 HR 池（scenario_uid）
    hr_uids = set()
    for fn in ('train_hr_labels.csv', 'val_hr_labels.csv'):
        p = os.path.join(e6.DATA_DIR, fn)
        hr_uids |= set(pd.read_csv(p, usecols=['scenario_uid'])
                       ['scenario_uid'].astype(str))
    out['hr_pools_988_180'] = {
        'n_pool': len(hr_uids),
        'uid_overlap': sorted(manifest_uids & hr_uids),
    }
    ok = ok_agent and all(
        not v.get('uid_overlap') and not v.get('env_id_overlap')
        for k, v in out.items() if k != 'agent_pools')
    return ok, out


def gen_scenarios(workers):
    os.makedirs(T1_DIR, exist_ok=True)
    t0 = time.time()
    tasks = [(SIMPLE_START_ID + i, 0.0) for i in range(N_PER_STRATUM)]
    tasks += [(COUPLED_START_ID + i, 1.0)
              for i in range(COUPLED_MAX_TRIES)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results, n_done = {}, 0
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(_gen_one, t): t for t in tasks}
        for fut in as_completed(futs):
            env_id, res = fut.result()
            results[env_id] = res
            n_done += 1
            if n_done % 10 == 0:
                log(f'  [gen] {n_done}/{len(tasks)} '
                    f'({time.time() - t0:.0f}s)')

    # ---- 分层选择（规则冻结：按 env_id 升序取前 20）----
    selected = []
    for i in range(N_PER_STRATUM):
        env_id = SIMPLE_START_ID + i
        res = results.get(env_id)
        assert res is not None, f'simple 流 env_id={env_id} 生成失败'
        assert int(res['meta']['neighbor_count']) == 0
        selected.append((env_id, 'simple', res))
    n_med = n_str = 0
    for i in range(COUPLED_MAX_TRIES):
        if n_med >= N_PER_STRATUM and n_str >= N_PER_STRATUM:
            break
        env_id = COUPLED_START_ID + i
        res = results.get(env_id)
        if res is None:
            continue
        diff = str(res['meta']['difficulty'])
        if diff == 'easy' and n_med < N_PER_STRATUM:
            selected.append((env_id, 'medium', res))
            n_med += 1
        elif diff in ('moderate', 'hard') and n_str < N_PER_STRATUM:
            selected.append((env_id, 'strong', res))
            n_str += 1
    assert n_med >= N_PER_STRATUM and n_str >= N_PER_STRATUM, \
        f'coupled 流 {COUPLED_MAX_TRIES} 个 env_id 内分层不足 ' \
        f'(medium={n_med}, strong={n_str})'
    assert len(selected) == 3 * N_PER_STRATUM
    log(f'[gen] 选中 {len(selected)} 场景 '
        f'(simple={N_PER_STRATUM}, medium={n_med}, strong={n_str}) '
        f'({time.time() - t0:.0f}s)')

    # ---- 落盘（复用 exp6 的落盘链路与逐场景 loader 验证）----
    gas = e6._import_from_path('generate_agent_scenarios', e6.AGENT_GEN_PATH)
    loader = gas._import_from_path('neighbor_scenario_loader',
                                   gas.LOADER_PATH)
    metas_out, manifest_entries, verify_fail = [], [], []
    for env_id, stratum, res in selected:
        meta = dict(res['meta'])
        meta['training_seen'] = False
        meta['agent_split'] = 't1_independent'
        meta['stratum'] = stratum
        metas_out.append(meta)
        csv_path = os.path.join(T1_DIR, f'scenario_{env_id:03d}_pareto.csv')
        pd.DataFrame(res['records']).to_csv(csv_path, index=False)
        sc_json = gas._meta_to_loader_scenario(meta, 't1_independent',
                                               T1_DIR)
        sc_json['stratum'] = stratum
        json_path = os.path.join(T1_DIR, f'scenario_{env_id:03d}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sc_json, f, ensure_ascii=False)
        try:
            loader.load_scenario(json_path, n_seg=N_SEG)
            verr = ''
        except Exception as exc:  # noqa: BLE001
            verr = str(exc)
        verify_fail.append((env_id, verr))
        manifest_entries.append({
            'env_id': int(env_id),
            'scenario_uid': str(meta['scenario_name']),
            'stratum': stratum,
            'neighbor_count': int(meta['neighbor_count']),
            'difficulty': str(meta['difficulty']),
            'random_seed': int(meta['random_seed']),
            'n_attempts': int(res['n_attempts']),
            'loader_verify_error': verr,
        })
    n_bad = sum(1 for _, v in verify_fail if v)
    log(f'[gen] loader 验证失败 {n_bad} 个'
        + (f'：{[e for e, v in verify_fail if v]}' if n_bad else ''))

    meta_path = os.path.join(T1_DIR, 'scenario_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metas_out, f, ensure_ascii=False, default=e6._json_default)

    # ---- 不相交机器核验 ----
    manifest_uids = {e['scenario_uid'] for e in manifest_entries}
    manifest_names = {str(m.get('scenario_name')) for m in metas_out}
    manifest_seeds = {int(m.get('random_seed')) for m in metas_out}
    manifest_env_ids = {e['env_id'] for e in manifest_entries}
    ok, detail = _disjointness_t1(manifest_uids, manifest_names,
                                  manifest_seeds, manifest_env_ids)

    manifest = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'T1 reviewer independent test scenarios (60, new seed '
                   'stream + new env_id range; never used for training / '
                   'state selection / thresholding)',
        'seed_base': SEED_BASE_T1,
        'max_gen_attempts': MAX_GEN_ATTEMPTS,
        'n_per_stratum': N_PER_STRATUM,
        'simple_start_id': SIMPLE_START_ID,
        'coupled_start_id': COUPLED_START_ID,
        'coupled_max_tries': COUPLED_MAX_TRIES,
        'generator': os.path.relpath(e6.GENERATOR_PATH, _REPO_ROOT),
        'disjointness_ok': bool(ok),
        'disjointness': detail,
        'scenarios': manifest_entries,
        'file_sha256': {},
    }
    for fn in sorted(os.listdir(T1_DIR)):
        if fn.startswith('scenario_') and os.path.isfile(
                os.path.join(T1_DIR, fn)):
            manifest['file_sha256'][fn] = e6._sha256_file(
                os.path.join(T1_DIR, fn))
    tmp = T1_MANIFEST + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, T1_MANIFEST)
    log(f'[gen] manifest -> {T1_MANIFEST}  disjointness_ok={ok}')
    if not ok:
        raise RuntimeError('不相交机器核验失败，见 manifest disjointness 段')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=32)
    args = p.parse_args(argv)
    return gen_scenarios(args.workers)


if __name__ == '__main__':
    sys.exit(main())
