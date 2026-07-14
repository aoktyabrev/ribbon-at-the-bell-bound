"""D-S2. Кросс-скан (k_f×N) + форма при жёсткой связи (см. DS2_prereg.md). ТОЛЬКО СЫРЬЁ.

Кросс-скан: θ=0, нечёт, k_f∈{×1,×4}, N∈{16,32,64,96}. Форма: E_raw(θ) 9 точек,
нечёт, N=32, k_f∈{×1,×2,×4} (полные таблицы (s,t) для CHSH). M=1200. Свежие seed'ы.
Флип/фиты/CHSH — ПОСЛЕ коммита (analysis_ds2.py). Результат → DS2_raw.json.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_ds2.py [--smoke]
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
THETAS = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi]
THETA_NAMES = ["0", "pi/8", "pi/4", "3pi/8", "pi/2", "5pi/8", "3pi/4", "7pi/8", "pi"]
M_REPL = 150 if SMOKE else 1200
STEPS = 1500 if SMOKE else 4000


def measure(sector_odd, params, theta, N, key):
    """Точка (N,θ,k_f): M реплик, релаксация+доохлаждение, полная таблица (s,t) по AX."""
    a, b = M.apparatus_axes_theta(theta)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, sector_odd, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    rej = float(np.asarray(r1).sum() + np.asarray(r2).sum())
    s, t, degen = M.classify_batch(u, a, b)
    sm, tm, _ = M.classify_batch(u, -a, -b)
    s, t, sm, tm, degen = [np.asarray(z) for z in (s, t, sm, tm, degen)]
    keep = ~degen
    sv, tv = s[keep], t[keep]
    nv = int(keep.sum())
    tab = dict(pp=int(((sv > 0) & (tv > 0)).sum()), pm=int(((sv > 0) & (tv < 0)).sum()),
               mp=int(((sv < 0) & (tv > 0)).sum()), mm=int(((sv < 0) & (tv < 0)).sum()),
               degen=int(degen.sum()))
    E = float(np.mean(sv * tv)) if nv else 0.0
    Emir = float(np.mean((sm * tm)[keep])) if nv else 0.0
    sigma = float(np.sqrt(max(1.0 - E * E, 1e-9) / max(nv, 1)))
    return dict(E_raw=E, A=abs(E), sigma=sigma, E_mirror=Emir, table=tab, n_valid=nv,
                rej_per_step=rej / (M_REPL * (STEPS + STEPS // 2)))


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260714 + 202)
    kf_cross = [1.0, 4.0]
    Ns_cross = [16, 32] if SMOKE else [16, 32, 64, 96]
    kf_form = [1.0, 2.0] if SMOKE else [1.0, 2.0, 4.0]
    thetas = ([0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi] if SMOKE else THETAS)
    thnames = (["0", "pi/4", "pi/2", "3pi/4", "pi"] if SMOKE else THETA_NAMES)
    print(f"[DS2 {tag}] M={M_REPL} cross k_f={kf_cross}×N{Ns_cross}; form k_f={kf_form}×{len(thetas)}θ")

    data = {"meta": dict(tag=tag, M=M_REPL, T_mid=T_MID, base=BASE, commit_base="a9cef7b",
                         kf_cross=kf_cross, Ns_cross=Ns_cross, kf_form=kf_form,
                         thetas=thnames, theta_vals=[float(t) for t in thetas],
                         note="СЫРЬЁ AX нечёт свежие seed; флип/фиты/CHSH — ПОСЛЕ коммита"),
            "cross": {}, "form": {}}

    # --- кросс-скан θ=0 ---
    for mult in kf_cross:
        for N in Ns_cross:
            key, sk = jr.split(key)
            params = dict(BASE, k_f=BASE["k_f"] * mult)
            r = measure(True, params, 0.0, N, sk)
            data["cross"][f"kf{mult}|N{N}"] = r
            print(f"  cross k_f×{mult} N{N}: A={r['A']:.4f}±{r['sigma']:.4f} rej={r['rej_per_step']:.1e}")

    # --- форма E(θ), N=32 ---
    for mult in kf_form:
        params = dict(BASE, k_f=BASE["k_f"] * mult)
        for theta, thn in zip(thetas, thnames):
            key, sk = jr.split(key)
            r = measure(True, params, theta, 32, sk)
            data["form"][f"kf{mult}|{thn}"] = r
            print(f"  form k_f×{mult} θ{thn:5s}: E={r['E_raw']:+.4f}±{r['sigma']:.4f} tab={r['table']}")

    with open(os.path.join(RES, "DS2_raw.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  СЫРЬЁ → {RES}/DS2_raw.json (далее: КОММИТ, потом analysis_ds2.py)")


if __name__ == "__main__":
    main()
