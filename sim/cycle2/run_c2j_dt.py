"""C2-J dt-арбитраж (addendum-3). ЯВНАЯ схема фазы D (build_minimizer, НЕ adaptive).
Сходимость бассейнов по dt: λ-спаренный flip s между уровнями dt, A, C_ee,
E_final-разброс. k_c×{64,256} (+×1 реф.), b=−a, T=0, M=256.
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

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
DT0, PHYS_T, N, M_REPL = 5e-3, 20.0, 32, 256


def relax_explicit(params, prep, dt, a, b):
    steps = int(round(PHYS_T/dt))
    mini = M.build_minimizer(params, lr=dt, freeze_w=False)
    x, u, rej = mini["run"](jr.PRNGKey(0), prep["x0"], prep["u0"], prep["X0"], prep["XL"],
                            a, b, jnp.zeros((steps,)))
    return x, u, float(np.asarray(rej).sum())/(M_REPL*steps), steps


def measure(x, u, a, b, params):
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen; nv = int(keep.sum())
    A = abs(float(np.mean((s*t)[keep]))) if nv else 0.0
    nA = jax.vmap(lambda uu: M.end_axis(uu[0]))(u); nB = jax.vmap(lambda uu: M.end_axis(uu[-1]))(u)
    cee = float(np.mean(np.sum(np.asarray(nA)*np.asarray(nB), -1)))
    Ef = np.asarray(jax.vmap(lambda xx, uu: M.e_meas(xx, uu, a, b, params))(x, u))
    return s, A, cee, float(Ef.std())


def _require():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: --prereg-commit <hash>.")
    return sys.argv[sys.argv.index("--prereg-commit")+1]


def dt_ladder(mult):
    lad = [DT0, DT0/4, DT0/16, DT0/64]
    if mult >= 256:
        lad = [DT0/4, DT0/16, DT0/64, DT0/256]   # dt0 неустойчив при ×256
    return lad


def main():
    commit = _require()
    print(f"[C2-J dt-арбитраж] backend={jax.default_backend()}")
    a, b = M.apparatus_axes_theta(np.pi)          # b=−a
    out = {"meta": dict(prereg_commit=commit, backend=jax.default_backend(), M=M_REPL,
                        b_eq_minus_a=True, phys_time=PHYS_T), "cells": {}}
    key = jr.PRNGKey(20260717 + 55)
    Espread = {}
    for mult in (1.0, 64.0, 256.0):
        params = dict(BASE, k_c=BASE["k_c"]*mult)
        prep = M.prep_dynamics(N, True, M_REPL, key)   # ОБЩИЙ prep (λ) по dt
        ladder = dt_ladder(mult)
        rows = []; prev_s = None
        for dt in ladder:
            x, u, rej, steps = relax_explicit(params, prep, dt, a, b)
            s, A, cee, estd = measure(x, u, a, b, params)
            flip = float(np.mean(s != prev_s)) if prev_s is not None else None
            rows.append(dict(dt=dt, steps=steps, A=A, C_ee=cee, rej_per_step=rej,
                             flip_vs_coarser=flip, E_std=estd))
            fs = f"{flip:.4f}" if flip is not None else "  ref"
            print(f"  k_c×{mult:.0f} dt={dt:.2e} ({steps} шаг): A={A:.4f} C_ee={cee:+.3f} "
                  f"flip={fs} rej={rej:.1e} E_std={estd:.3f}")
            prev_s = s
        Espread[f"kc{mult}"] = rows[-1]["E_std"]
        out["cells"][f"kc{mult}"] = dict(kc=mult, ladder=[float(x) for x in ladder], rows=rows)
    out["E_std_ref_kc1"] = Espread.get("kc1.0")
    with open(os.path.join(RES, "C2J_dt_arbitrage.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2J_dt_arbitrage.json")


if __name__ == "__main__":
    main()
