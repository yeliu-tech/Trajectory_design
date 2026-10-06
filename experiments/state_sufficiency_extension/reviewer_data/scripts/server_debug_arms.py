# -*- coding: utf-8 -*-
"""server_debug_arms.py — 服务器上逐个策略臂直调，抓真实异常。"""
from __future__ import annotations

import os
import sys
import traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir,
                                          os.pardir, os.pardir))
for _p in (_REPO_ROOT,
           os.path.join(_REPO_ROOT, 'experiments',
                        'history_conditioned_ftg_final', 'scripts')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import hr_completion_oracle as orc  # noqa: E402
import exp2_relabel_train_val as e2r  # noqa: E402
import exp5_beam_analysis as e5  # noqa: E402

exp1 = orc._import_module(orc.EXP1_SCRIPT, 'exp1_vp')
pools = e2r.load_scenario_pools()
uid = 'nw_dual_val_001'
sc = pools[uid]
_rec, _steps, states = e5._prepare([uid], 9)
sha = sorted(states[uid].keys())[0]
prefix = np.asarray(states[uid][sha], dtype=np.float64)
seed, _ = orc.content_seed(uid, prefix, salt=e2r.SEED_SALT)

counter = {'n': 0}
for name, fn in (('sobol', lambda: exp1._sobol_completions(sc, prefix, 100, seed, counter)),
                 ('slsqp', lambda: exp1._slsqp_completions(sc, prefix, 50, None, counter)),
                 ('nsga', lambda: orc._nsga2_completions_fixed_pop(exp1, sc, prefix, 50, seed, counter))):
    try:
        comps = fn()
        print(f'[ok] {name}: {len(comps)} completions', flush=True)
    except Exception:
        print(f'[FAIL] {name}:', flush=True)
        traceback.print_exc()
        break
