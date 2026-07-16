"""C2-A — аудит якорей A(k_f) (prereg 2535c8b). STAGED.
Один процесс = один бэкенд (JAX platform глобален). Пишет C2A_raw_<backend>.json.

--backend gpu : все 4 якоря k_f×{0.5,1,2,4} × 3 свежих seed (GPU fp64).
--backend cpu : k_f×1 × 3 свежих seed (JAX_PLATFORMS=cpu, тот же код).
Протокол S1-R1: нечёт, θ=0, N=32, M=1200, T_mid=0.05 + доохлаждение.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2a.py --backend {gpu|cpu} --prereg-commit <hash>
"""
import os
import sys

BACKEND = "cpu" if ("--backend" in sys.argv and sys.argv[sys.argv.index("--backend")+1] == "cpu") else "gpu"
if BACKEND == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR, T_MID, N, M_REPL, STEPS = 5e-3, 0.05, 32, 1200, 4000
SEEDS = [11, 22, 33]                    # 3 свежих seed


def measure(mult, key):
    params = dict(BASE, k_f=BASE["k_f"] * mult)
    a, b = M.apparatus_axes_theta(0.0)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen; nv = int(keep.sum())
    E = float(np.mean((s * t)[keep])) if nv else 0.0
    return dict(A=abs(E), sigma_bin=float(np.sqrt(max(1 - E*E, 1e-9)/max(nv, 1))),
                n_valid=nv, degen=int(degen.sum()))


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash>.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    mults = [0.5, 1.0, 2.0, 4.0] if BACKEND == "gpu" else [1.0]
    dev = jax.devices()
    meta = dict(script="run_c2a", prereg_commit=commit, backend=jax.default_backend(),
                jax_version=jax.__version__, device=str(dev[0]), x64=jax.config.jax_enable_x64,
                N=N, M=M_REPL, T_mid=T_MID, seeds=SEEDS, protocol="a9cef7b S1-R1")
    print(f"[C2-A {BACKEND}] backend={meta['backend']} device={meta['device']} mults={mults}")
    out = {"meta": meta, "cells": {}}
    for mult in mults:
        A_list = []
        for sd in SEEDS:
            r = measure(mult, jr.PRNGKey(20260716 + 500 + sd))
            A_list.append(r["A"])
            print(f"  k_f×{mult} seed={sd}: A={r['A']:.4f}±{r['sigma_bin']:.4f} degen={r['degen']}")
        A_arr = np.array(A_list)
        out["cells"][f"kf{mult}"] = dict(A=[float(x) for x in A_arr], mean=float(A_arr.mean()),
                                         s_seed=float(A_arr.std(ddof=1)))
        print(f"    k_f×{mult}: μ={A_arr.mean():.4f} s_seed={A_arr.std(ddof=1):.4f}")
    with open(os.path.join(RES, f"C2A_raw_{BACKEND}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2A_raw_{BACKEND}.json")


if __name__ == "__main__":
    main()
