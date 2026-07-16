"""C2-ISO-DYN сокращённая кампания (prereg 77f734c). STAGED.
Динамическая изотропизация: пер-репличный Haar-поворот приготовления
(haar_rotate_prep), лаб-фиксированные (a,b) в релаксации И считывании.
θ∈{0,π/4,π/2,3π/4,π}, 4 CHSH-коррелятора, k_f∈{×1,×4}, N=32, M=4800, GPU fp64.
БЕЗ изотропной формулы; ВСЕ события, sign(0)→+1; DEGENERATE колонкой (не исключ.).

Валидация T-cov (в составе prereg) — гейт до кампании: энергетическая
инвариантность e_meas под (prep,a,b)→(R·prep, R·a, R·b), и no-op тождества.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2iso.py --prereg-commit <hash>
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
from ribbon_sim.frames import rotmat
import coupling as C

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

SMOKE = "--smoke" in sys.argv
BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR, T_MID, N = 5e-3, 0.05, 32
M_REPL = 8 if SMOKE else 4800
STEPS = 12 if SMOKE else 4000
THETAS = {"0": 0.0, "pi/4": np.pi/4, "pi/2": np.pi/2, "3pi/4": 3*np.pi/4, "pi": np.pi}


def axis(theta):
    return jnp.array([jnp.sin(theta), 0.0, jnp.cos(theta)])


def relax(mini, prep, key, a, b):
    sched = jnp.concatenate([jnp.full((STEPS,), T_MID), jnp.zeros((STEPS // 2,))])
    _, u, _ = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u


def Ecorr(u, a, b):
    """E=⟨s·t⟩ по ВСЕМ событиям (sign(0)→+1), degen — колонкой (не исключается)."""
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    st = s * t
    E = float(np.mean(st))
    return E, float(np.sqrt(max(1 - E*E, 1e-9) / st.size)), float(np.mean(degen))


def validate_tcov(prep):
    """T-cov гейт (энергетический, детерминированный). KILL при провале."""
    q = C.haar_quats(jr.PRNGKey(7), prep["u0"].shape[0])
    Rm = jax.vmap(rotmat)(q)
    prep_R = C.haar_rotate_prep(prep, q)
    a, b = M.apparatus_axes_theta(0.7)
    e_ref = jax.vmap(lambda x, u: M.e_meas(x, u, a, b, BASE))(prep["x0"], prep["u0"])
    e_rot = jax.vmap(lambda x, u, R: M.e_meas(x, u, R @ a, R @ b, BASE))(prep_R["x0"], prep_R["u0"], Rm)
    tcov1 = float(jnp.max(jnp.abs(e_ref - e_rot)))
    q_id = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (prep["u0"].shape[0], 1))
    pr_id = C.haar_rotate_prep(prep, q_id)
    tcov2 = float(jnp.max(jnp.abs(pr_id["x0"] - prep["x0"])))
    return tcov1, tcov2


def run_cell(kf, key):
    params = dict(BASE, k_f=BASE["k_f"] * kf)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    prep = M.prep_dynamics(N, True, M_REPL, key)
    q = C.haar_quats(jr.fold_in(key, 99), M_REPL)
    prep_R = C.haar_rotate_prep(prep, q)     # изотропизованное приготовление
    a0 = axis(0.0)
    theta = {}
    for name, th in THETAS.items():
        u = relax(mini, prep_R, key, a0, axis(th))
        E, sig, deg = Ecorr(u, a0, axis(th))
        theta[name] = dict(theta=float(th), E=E, sigma=sig, degen=deg)
        print(f"  k_f×{kf} θ={name:5s}: E={E:+.4f}±{sig:.4f} degen={deg:.3f}")
    # CHSH: a∈{0,π/2}, b∈{π/4,3π/4}; e00,e01 = θ-grid π/4,3π/4 (reuse)
    e00 = theta["pi/4"]["E"]; s00 = theta["pi/4"]["sigma"]
    e01 = theta["3pi/4"]["E"]; s01 = theta["3pi/4"]["sigma"]
    u10 = relax(mini, prep_R, key, axis(np.pi/2), axis(np.pi/4)); e10, s10, _ = Ecorr(u10, axis(np.pi/2), axis(np.pi/4))
    u11 = relax(mini, prep_R, key, axis(np.pi/2), axis(3*np.pi/4)); e11, s11, _ = Ecorr(u11, axis(np.pi/2), axis(3*np.pi/4))
    S = e00 - e01 + e10 + e11
    sS = float(np.sqrt(s00**2 + s01**2 + s10**2 + s11**2))
    print(f"  k_f×{kf} CHSH |S|={abs(S):.4f}±{sS:.4f}  {'S≤2 ✓' if abs(S)<=2+sS else 'S>2 — АУДИТ!'}")
    return dict(kf=kf, theta=theta, chsh=dict(S=float(S), sigma=sS,
                E=dict(e00=e00, e01=e01, e10=e10, e11=e11)))


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash>.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2iso", prereg_commit=commit, backend=jax.default_backend(),
                jax_version=jax.__version__, device=str(dev[0]), x64=jax.config.jax_enable_x64,
                N=N, M=M_REPL, T_mid=T_MID, grid=list(THETAS))
    print(f"[C2-ISO-DYN] backend={meta['backend']} device={meta['device']} M={M_REPL}")
    # T-cov гейт
    prep0 = M.prep_dynamics(N, True, 64, jr.PRNGKey(1))
    t1, t2 = validate_tcov(prep0)
    print(f"  T-cov-1 (энерг. инвариант) max|Δe|={t1:.2e}; T-cov-2 (id no-op) max|Δ|={t2:.2e}")
    if t1 > 1e-9 or t2 > 1e-12:
        sys.exit(f"KILL валидации T-cov: t1={t1:.2e} t2={t2:.2e} — кампания не стартует.")
    meta["tcov1"] = t1; meta["tcov2"] = t2
    out = {"meta": meta, "cells": {}}
    key = jr.PRNGKey(20260716 + 700)
    for kf in (1.0, 4.0):
        key, sk = jr.split(key)
        out["cells"][f"kf{kf}"] = run_cell(kf, sk)
    with open(os.path.join(RES, "C2ISO_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2ISO_raw.json (далее: КОММИТ, потом анализ)")


if __name__ == "__main__":
    main()
