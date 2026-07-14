"""D-S3. Карта анизотропии + источник для изотропизации (см. DS3_prereg.md). СЫРЬЁ.

Часть 1: E(a,b) при наклоне a на α от ê, b=−a и b под π/4 (зажим-во-время-релаксации).
Часть 2: источник — релаксация нечёт при зажимах вдоль ê (θ_src=0), сохраняем осевые
векторы концов n_A,n_B (для изотропизации в analysis_ds3, честная shared randomness).
Флип/фиты/R-усреднение/CHSH — ПОСЛЕ коммита. Результат → DS3_raw.json.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_ds3.py [--smoke]
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
LR, T_MID, N = 5e-3, 0.05, 32
STEPS = 1500 if SMOKE else 4000
M1 = 150 if SMOKE else 1200       # карта анизотропии
M2 = 300 if SMOKE else 2400       # источник изотропизации
KF = [1.0] if SMOKE else [1.0, 4.0]
ALPHAS = [0.0, np.pi/4, np.pi/2] if SMOKE else [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
ANAMES = ["0", "pi/4", "pi/2"] if SMOKE else ["0", "pi/8", "pi/4", "3pi/8", "pi/2"]

_end_axis_b = jax.vmap(lambda uu: jnp.stack([M.end_axis(uu[0]), M.end_axis(uu[-1])]))


def relax(params, a, b, mrep, key):
    """Релаксация с зажимами (a,b). Возврат финальные u (mrep,N,4)."""
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, mrep, key)
    k1, k2 = jr.split(key)
    x, u, _ = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, _ = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    return u


def corr(u, a, b):
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen
    E = float(np.mean((s * t)[keep])) if keep.sum() else 0.0
    sig = float(np.sqrt(max(1 - E*E, 1e-9) / max(keep.sum(), 1)))
    return E, sig, int((~keep).sum())


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260714 + 404)
    print(f"[DS3 {tag}] N={N} M1={M1} M2={M2} k_f={KF}")
    data = {"meta": dict(tag=tag, N=N, M1=M1, M2=M2, KF=KF, alphas=ANAMES,
                         alpha_vals=[float(a) for a in ALPHAS], base=BASE,
                         commit_base="f928dd4", note="СЫРЬЁ; фиты/R-усреднение — ПОСЛЕ коммита"),
            "aniso": {}, "source": {}}

    # ===== Часть 1: карта анизотропии =====
    for mult in KF:
        params = dict(BASE, k_f=BASE["k_f"] * mult)
        for al, an in zip(ALPHAS, ANAMES):
            # b = −a (антипараллель, θ=π)
            key, sk = jr.split(key)
            a, b = M.apparatus_axes_ab(al, al + np.pi)
            u = relax(params, a, b, M1, sk)
            E, sig, deg = corr(u, a, b)
            data["aniso"][f"kf{mult}|a{an}|anti"] = dict(E=E, sigma=sig, degen=deg)
            # b под θ=π/4 к a
            key, sk = jr.split(key)
            a, b = M.apparatus_axes_ab(al, al + np.pi/4)
            u = relax(params, a, b, M1, sk)
            E2, sig2, deg2 = corr(u, a, b)
            data["aniso"][f"kf{mult}|a{an}|pi4"] = dict(E=E2, sigma=sig2, degen=deg2)
            print(f"  aniso k_f×{mult} α={an:5s}: E_anti={E:+.3f}±{sig:.3f}  E_pi4={E2:+.3f}±{sig2:.3f}")

    # ===== Часть 2: источник (зажим вдоль ê, θ_src=0), сохранить n_A,n_B =====
    for mult in KF:
        params = dict(BASE, k_f=BASE["k_f"] * mult)
        key, sk = jr.split(key)
        a, b = M.apparatus_axes_theta(0.0)   # обе оси = ê
        u = relax(params, a, b, M2, sk)
        n = np.asarray(_end_axis_b(u))        # (M2, 2, 3): [:,0]=n_A, [:,1]=n_B
        data["source"][f"kf{mult}"] = dict(n_A=n[:, 0].tolist(), n_B=n[:, 1].tolist())
        # диагностика: сырой E при a=b=ê
        E, sig, deg = corr(u, a, b)
        print(f"  source k_f×{mult}: M2={M2} n_A,n_B сохранены; сырой E(θ=0)={E:+.3f}±{sig:.3f}")

    with open(os.path.join(RES, "DS3_raw.json"), "w") as f:
        json.dump(data, f)
    print(f"  СЫРЬЁ → {RES}/DS3_raw.json (далее: КОММИТ, потом analysis_ds3.py)")


if __name__ == "__main__":
    main()
