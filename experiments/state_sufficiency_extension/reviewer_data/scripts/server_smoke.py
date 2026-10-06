# -*- coding: utf-8 -*-
"""server_smoke.py — 服务器(server-210)迁移后冒烟测试。

验证项：
  1. 关键模块可导入（hr_completion_oracle / features_v3 /
     exp2_relabel_train_val / generate_training_scenarios）；
  2. 冻结 HGBR 模型 model_m0/model_m1 可加载且 sha256 与冻结值一致；
  3. 端到端：对 val 池 1 个场景 + 1 个 prefix 跑一次 production
     tier=200 evaluate_prefix，打印 J_star_cont 与耗时。

用法（服务器上，cwd=project root）：
  /home/eg840/LiuProject/rl_trajectory/venv/bin/python \
      experiments/state_sufficiency_extension/reviewer_data/scripts/server_smoke.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir,
                                          os.pardir, os.pardir))
_HCFTG_SCRIPTS = os.path.join(_REPO_ROOT, 'experiments',
                              'history_conditioned_ftg_final', 'scripts')
_RL_RETRAIN = os.path.join(_REPO_ROOT, 'neighboring_well_benchmark_v1',
                           'rl_retrain')
for _p in (_REPO_ROOT, _HCFTG_SCRIPTS, _RL_RETRAIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

M0_SHA256 = ('1e93c4b2af36b201c1fe3c9ad85b0a17f12e180db008ab1e53a6bc'
             'dbfdbb8fe8')
M1_SHA256 = ('c3ca6bb3c0c4f4dd7e4f5b6f0a398f47ef10a2f97347d824d371e2'
             'fe285c2b9a')


def log(msg):
    print(msg, flush=True)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    import numpy as np
    import joblib

    log(f'[env] python {sys.version.split()[0]}')
    import numpy, pandas, scipy, sklearn, pymoo
    log(f'[env] numpy {numpy.__version__} pandas {pandas.__version__} '
        f'scipy {scipy.__version__} sklearn {sklearn.__version__} '
        f'pymoo {pymoo.__version__}')

    # 1. 模块导入
    import hr_completion_oracle as orc
    log('[ok] import hr_completion_oracle')
    import features_v3
    log('[ok] import features_v3')
    import exp2_relabel_train_val as e2r
    log('[ok] import exp2_relabel_train_val')
    import generate_training_scenarios as gen
    log('[ok] import generate_training_scenarios')

    # 2. 冻结模型加载 + 指纹
    mdir = os.path.join(_REPO_ROOT, 'experiments',
                        'history_conditioned_ftg_final', 'models', 'exp21')
    for name, ref in (('model_m0.joblib', M0_SHA256),
                      ('model_m1.joblib', M1_SHA256)):
        p = os.path.join(mdir, name)
        h = sha256(p)
        assert h == ref, f'{name} sha256 不匹配: {h[:12]} != {ref[:12]}'
        m = joblib.load(p)
        log(f'[ok] {name} sha256={h[:12]} members={len(m["members"])} '
            f'n_features={len(m["features"])}')

    # 3. 端到端：1 个 val 场景 + 1 个 prefix，production tier=200
    exp1 = orc._import_module(orc.EXP1_SCRIPT, 'exp1_vp')
    from scripts.phase32.p32m_engineering_review import review_trajectory
    pools = e2r.load_scenario_pools()
    import exp5_beam_analysis as e5
    uid = sorted(e5.load_val_meta()['scenario_uid'].unique())[0]
    sc = pools[uid]
    log(f'[e2e] scenario={uid}')

    _rec, _steps, states = e5._prepare([uid], 9)
    sha = sorted(states[uid].keys())[0]
    prefix = np.asarray(states[uid][sha], dtype=np.float64)
    log(f'[e2e] prefix k={len(prefix)//3} sha={sha[:12]}')

    seed, _ = orc.content_seed(uid, prefix, salt=e2r.SEED_SALT)
    n_sobol, n_slsqp, n_nsga = orc.budget_split(200)
    t0 = time.perf_counter()
    _recs, agg = orc.evaluate_prefix(
        exp1, sc, prefix, seed, n_sobol, n_slsqp, n_nsga,
        review_trajectory, nsga_mode='nested')
    wall = time.perf_counter() - t0
    log(f'[e2e] tier=200 J_star_cont={agg["J_star_cont"]:.6f} '
        f'evals={agg["evals_total"]} wall={wall:.1f}s '
        f'n_errors={agg.get("n_errors", 0)}')
    log('SMOKE_PASS')


if __name__ == '__main__':
    main()
