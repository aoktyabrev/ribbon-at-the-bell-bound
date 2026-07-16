"""C2-F0 — гейт хаоса (см. prereg_drafts/C2F_prereg_DRAFT.md, стадия F0).
STAGED: НЕ запускать до утверждения Артемом и prereg-коммита (после C2-V0 PASS).

Нуль-тест γ=0 (побитовый) → лестница F(δ), δ ∈ {1e−6,1e−4,1e−2,π/8,π/2}, для
N=32, k_f×4, M=1200, в двух T-ячейках {0.05, 0}. Сырьё → C2F0_raw.json;
ветвление (проход/плато/серая зона) — в анализе ПОСЛЕ коммита, s_min от Артема.

Запуск (после утверждения): PYTHONPATH=src:phase_D:cycle2 \
    python cycle2/run_c2f0.py --prereg-commit <hash>
Бэкенд GPU/fp64. Наблюдаемые — через cycle2.coupling (образец iso_corr).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M
import coupling as C

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR, N, M_REPL, STEPS = 5e-3, 32, 1200, 4000
KF_MULT = 4.0
DELTAS = [1e-6, 1e-4, 1e-2, np.pi/8, np.pi/2]
T_CELLS = [0.05, 0.0]


def _prep_and_mini(params, key):
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, M_REPL, key)
    return mini, p


def _schedule(T):
    """Релаксация при T + доохлаждение (T=0 квенч ⇒ всё расписание при 0)."""
    return jnp.concatenate([jnp.full((STEPS,), T), jnp.full((STEPS // 2,), 0.0)])


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash> после утверждения Артемом.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2f0", prereg_commit=commit,
                backend=jax.default_backend(), jax_version=jax.__version__,
                device=str(dev[0]), x64=jax.config.jax_enable_x64,
                N=N, M=M_REPL, kf_mult=KF_MULT, deltas=DELTAS, T_cells=T_CELLS)
    print(f"[C2-F0] backend={meta['backend']} device={meta['device']} k_f×{KF_MULT}")
    params = dict(BASE, k_f=BASE["k_f"] * KF_MULT)
    a, _ = M.apparatus_axes_theta(0.0)        # a=ê фиксирована
    out = {"meta": meta, "cells": {}}
    key = jr.PRNGKey(20260716 + 40)

    for T in T_CELLS:
        key, sk = jr.split(key)
        mini, prep = _prep_and_mini(params, sk)
        sched = _schedule(T)
        # b базовая = ê; b'(δ) = поворот b на δ в плоскости (x,z)
        b0 = a
        # нуль-тест γ=0: побитовый
        diff, ok0 = C.null_test_gamma0(mini, prep, sk, a, b0, sched)
        cell = {"null_gamma0_maxdiff": diff, "null_ok": bool(ok0), "F": {}}
        if not ok0:
            print(f"  T={T}: НУЛЬ-ТЕСТ ПРОВАЛЕН (max|Δ|={diff:.1e}) — баг раздачи ключей, стоп")
            out["cells"][f"T{T}"] = cell
            continue
        for d in DELTAS:
            _, bp = M.apparatus_axes_theta(d)
            u_b, u_bp = C.coupled_run(mini, prep, sk, a, b0, bp, sched)
            F, sigF, deg = C.flip_rate(u_b, u_bp, a, b0, bp)
            Dlt, pb, pbp = C.marginal_gap(u_b, u_bp, a, b0, bp)
            cell["F"][f"{d:.3e}"] = dict(F=F, sigma=sigF, degen_diag=deg,
                                         Delta=Dlt, P_plus_b=pb, P_plus_bp=pbp)
            print(f"  T={T} δ={d:.1e}: F={F:.4f}±{sigF:.4f}  Δ={Dlt:.4f}  degen={deg:.3f}")
        out["cells"][f"T{T}"] = cell

    with open(os.path.join(RES, "C2F0_raw.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  СЫРЬЁ → {RES}/C2F0_raw.json (далее: КОММИТ, потом анализ ветвления F0)")


if __name__ == "__main__":
    main()
