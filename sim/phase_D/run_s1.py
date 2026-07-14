"""S1-runs. Пост-D прогоны для дискриминации S1 (см. S1runs_prereg.md). ТОЛЬКО СЫРЬЁ.

R1 A∞(k_f), R2 A∞(T), R3 чёт vs нечёт — при θ=0, N=32, M=1200. E_raw=⟨s·t⟩ по AX.
Флип не нужен (|E|). Результат → S1runs_raw.json; таблицы/разности — ПОСЛЕ коммита.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_s1.py [--smoke]
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

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR = 5e-3
T_MID = 0.05
N = 16 if SMOKE else 32
M_REPL = 150 if SMOKE else 1200
STEPS = 1500 if SMOKE else 4000


def measure(sector_odd, params, T, key):
    """Точка θ=0: M реплик, релаксация T + доохлаждение, A=|⟨s·t⟩| по AX + зеркало."""
    a, b = M.apparatus_axes_theta(0.0)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, sector_odd, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    rej = float(np.asarray(r1).sum() + np.asarray(r2).sum())
    s, t, degen = M.classify_batch(u, a, b)
    sm, tm, _ = M.classify_batch(u, -a, -b)
    s, t, sm, tm, degen = [np.asarray(z) for z in (s, t, sm, tm, degen)]
    keep = ~degen
    sv, tv = s[keep], t[keep]
    n_valid = int(keep.sum())
    E = float(np.mean((sv * tv))) if n_valid else 0.0
    Emir = float(np.mean((sm * tm)[keep])) if n_valid else 0.0
    P_al = float(np.mean(((sv * tv) > 0))) if n_valid else 0.0
    sigma = float(np.sqrt(max(1.0 - E * E, 1e-9) / max(n_valid, 1)))
    return dict(E_raw=E, A=abs(E), sigma=sigma, E_mirror=Emir, P_aligned=P_al,
                n_valid=n_valid, degen=int(degen.sum()),
                rej_per_step=rej / (M_REPL * (STEPS + STEPS // 2)))


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260714 + 101)
    print(f"[S1-runs {tag}] N={N} M={M_REPL} θ=0")
    data = {"meta": dict(tag=tag, N=N, M=M_REPL, T_mid=T_MID, base=BASE,
                         commit_base="28db20d", note="СЫРЬЁ AX θ0; анализ — ПОСЛЕ коммита"),
            "R1_kf": {}, "R2_T": {}, "R3_sector": {}}

    # R1: A∞(k_f), нечёт, T_mid
    for mult in ([0.5, 1.0] if SMOKE else [0.5, 1.0, 2.0, 4.0]):
        key, sk = jr.split(key)
        params = dict(BASE, k_f=BASE["k_f"] * mult)
        r = measure(True, params, T_MID, sk)
        data["R1_kf"][f"kf{mult}"] = r
        print(f"  R1 k_f×{mult}: A={r['A']:.4f}±{r['sigma']:.4f} P_al={r['P_aligned']:.3f} "
              f"rej={r['rej_per_step']:.1e} degen={r['degen']}")

    # R2: A∞(T), нечёт, k_f=base
    for mult in [0.5, 1.0, 2.0]:
        key, sk = jr.split(key)
        r = measure(True, BASE, T_MID * mult, sk)
        data["R2_T"][f"T{T_MID*mult:.3f}"] = r
        print(f"  R2 T={T_MID*mult:.3f}: A={r['A']:.4f}±{r['sigma']:.4f} P_al={r['P_aligned']:.3f} "
              f"rej={r['rej_per_step']:.1e}")

    # R3: чёт vs нечёт, T_mid, k_f=base
    for sector_odd, sname in [(False, "even"), (True, "odd")]:
        key, sk = jr.split(key)
        r = measure(sector_odd, BASE, T_MID, sk)
        data["R3_sector"][sname] = r
        print(f"  R3 {sname}: A={r['A']:.4f}±{r['sigma']:.4f} P_al={r['P_aligned']:.3f}")

    with open(os.path.join(RES, "S1runs_raw.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  СЫРЬЁ → {RES}/S1runs_raw.json (далее: КОММИТ, потом анализ)")


if __name__ == "__main__":
    main()
