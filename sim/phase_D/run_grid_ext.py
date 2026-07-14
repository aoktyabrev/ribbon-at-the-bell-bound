"""D2-ext. Целевое разрешение скейлинга A_N (см. D2ext_prereg.md).

Нечётный сектор, θ∈{0,π/4}, N∈{16,32,48,64,96}, M=1200, T_mid. ТОЛЬКО СЫРЬЁ:
E_raw=⟨s·t⟩ по AX, σ, зеркала. Флип/фиты — ПОСЛЕ коммита (analysis_ext.py).
Точки N∈{16,32,48} пересчитываются на M=1200 (НЕ смешивать с D2_raw_tables.json).

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_grid_ext.py [--smoke]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR = 5e-3
T_MID = 0.05
THETAS = [(0.0, "0"), (np.pi / 4, "pi/4")]
if SMOKE:
    NS = [16, 32]
    M_REPL = 200
    STEPS = 1500
else:
    NS = [16, 32, 48, 64, 96]
    M_REPL = 1200
    STEPS = 4000


def measure(N, theta, key):
    """Нечёт, точка (N,θ): M реплик, релаксация+доохлаждение, таблица (s,t) по AX.
    Возврат E_raw, σ, таблица, rej, а также зеркальные E."""
    a, b = M.apparatus_axes_theta(theta)
    mini = M.build_minimizer(PARAMS, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, M_REPL, key)   # нечётный сектор
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    rej = float(np.asarray(r1).sum() + np.asarray(r2).sum())
    s, t, degen = M.classify_batch(u, a, b)
    sm, tm, _ = M.classify_batch(u, -a, -b)   # зеркало (a,b)→(−a,−b)
    s, t, sm, tm, degen = [np.asarray(z) for z in (s, t, sm, tm, degen)]
    keep = ~degen
    sv, tv = s[keep], t[keep]
    tab = dict(pp=int(((sv > 0) & (tv > 0)).sum()), pm=int(((sv > 0) & (tv < 0)).sum()),
               mp=int(((sv < 0) & (tv > 0)).sum()), mm=int(((sv < 0) & (tv < 0)).sum()),
               degen=int(degen.sum()))
    n_valid = tab["pp"] + tab["pm"] + tab["mp"] + tab["mm"]
    E = float((tab["pp"] + tab["mm"] - tab["pm"] - tab["mp"]) / max(n_valid, 1))
    Emir = float(np.mean((sm * tm)[keep]))
    sigma = float(np.sqrt(max(1.0 - E * E, 1e-9) / max(n_valid, 1)))
    return dict(E_raw=E, sigma=sigma, table=tab, n_valid=n_valid, E_mirror=Emir,
                rej_per_step=rej / (M_REPL * (STEPS + STEPS // 2)))


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260714 + 7)
    print(f"[grid-ext {tag}] нечёт, N={NS}, M={M_REPL}, θ={{0,π/4}}, T={T_MID}")
    data = {"meta": dict(tag=tag, NS=NS, M=M_REPL, thetas=["0", "pi/4"], T_mid=T_MID,
                         params=PARAMS, commit_base="9d7b8e9",
                         note="СЫРЬЁ AX нечёт M=1200; флип/фиты — ПОСЛЕ коммита"),
            "odd": {}}
    for N in NS:
        for theta, thn in THETAS:
            key, sk = jr.split(key)
            r = measure(N, theta, sk)
            cell = f"N{N}|odd|{thn}"
            data["odd"][cell] = r
            print(f"  {cell:14s}: E_raw={r['E_raw']:+.4f}±{r['sigma']:.4f} "
                  f"E_mir={r['E_mirror']:+.4f} rej={r['rej_per_step']:.1e} degen={r['table']['degen']}")
    with open(os.path.join(RES, "D2ext_raw_tables.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  СЫРЬЁ → {RES}/D2ext_raw_tables.json (далее: КОММИТ, потом analysis_ext.py)")


if __name__ == "__main__":
    main()
