"""C2-R — граница режима κ* (prereg ddc9880). STAGED.
--stage r0     : R0-гейт (якорь A(k_f×4) + rej_per_step лестницы). Быстро.
--stage ladder : лестница k_f×{8,32,128,512}, N∈{8,32}: F_s (H-R1/R2) + форма
                 H-R4a (θ-сетка 9т + CHSH + per-replica R для H-R6). GPU fp64 T=0.
lr-правило: lr = lr0·min(1, 4/k_f_mult) (Q1). σ_pos=const (Q5). M=4800.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2r.py --stage {r0|ladder} --prereg-commit <hash>
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

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR0, T_MID = 5e-3, 0.05
M_REPL = 8 if SMOKE else 4800
STEPS = 12 if SMOKE else 4000
REF = M.REF_AXIS
LADDER = [8.0, 32.0, 128.0, 512.0]
THETAS9 = [i*np.pi/8 for i in range(9)]
A_ANCHOR, S_SEED = 0.839, 0.024


def lr_for(mult):
    return LR0 * min(1.0, 4.0/mult)


def axis(theta):
    return jnp.array([jnp.sin(theta), 0.0, jnp.cos(theta)])


def relax(mini, prep, key, a, b, T):
    sched = jnp.concatenate([jnp.full((STEPS,), T), jnp.zeros((STEPS//2,))])
    _, u, rej = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u, float(np.asarray(rej).sum())


def measure_A(mult, key):
    """S1-протокол A(k_f) с lr-правилом."""
    mini = M.build_minimizer(dict(BASE, k_f=mult), lr=lr_for(mult), freeze_w=False)
    prep = M.prep_dynamics(32, True, 1200 if not SMOKE else 8, key)
    a, b = M.apparatus_axes_theta(0.0)
    u, rej = relax(mini, prep, key, a, b, T_MID)
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen; nv = int(keep.sum())
    E = float(np.mean((s*t)[keep])) if nv else 0.0
    return abs(E), float(np.sqrt(max(1-E*E, 1e-9)/max(nv, 1))), rej/(1200*(STEPS+STEPS//2))


def stage_r0(commit):
    print("[C2-R R0] якорь + rej-лестница")
    key = jr.PRNGKey(20260716 + 900)
    A4, s4, rej4 = measure_A(4.0, key)          # lr не скейлится при ×4
    scomb = float(np.sqrt(s4**2 + S_SEED**2))
    anchor_ok = abs(A4 - A_ANCHOR) <= 3*scomb
    print(f"  якорь A(k_f×4)={A4:.4f}±{s4:.4f} vs {A_ANCHOR} ⇒ {abs(A4-A_ANCHOR)/scomb:.2f}σ_comb "
          f"[{'OK' if anchor_ok else 'KILL'}]")
    rej_ok = True; rej_tab = {}
    for mult in LADDER:
        key, sk = jr.split(key)
        mini = M.build_minimizer(dict(BASE, k_f=mult), lr=lr_for(mult), freeze_w=False)
        prep = M.prep_dynamics(32, True, 1200 if not SMOKE else 8, sk)
        a, b = M.apparatus_axes_theta(0.0)
        _, rej = relax(mini, prep, sk, a, b, 0.0)
        rps = rej/((1200 if not SMOKE else 8)*(STEPS+STEPS//2))
        rej_tab[f"kf{mult}"] = rps
        ok = rps < 1e-3; rej_ok = rej_ok and ok
        print(f"  k_f×{mult} lr={lr_for(mult):.2e}: rej_per_step={rps:.2e} [{'OK' if ok else 'KILL'}]")
    passed = anchor_ok and rej_ok
    out = dict(meta=dict(script="run_c2r_r0", prereg_commit=commit, backend=jax.default_backend()),
               anchor=dict(A=A4, sigma=s4, ref=A_ANCHOR, sigma_comb=scomb, ok=bool(anchor_ok)),
               rej_ladder=rej_tab, R0_pass=bool(passed))
    with open(os.path.join(RES, "C2R_R0.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2R_R0.json  R0-ГЕЙТ: {'PASS' if passed else 'KILL — рельс сорван, стоп-отчёт'}")
    if not passed:
        sys.exit(1)


def signs(u):
    s, t, degen = M.classify_batch(u, REF, REF)
    return np.asarray(s), np.asarray(t), np.asarray(degen)


def stage_ladder(commit):
    dev = jax.devices()
    meta = dict(script="run_c2r_ladder", prereg_commit=commit, backend=jax.default_backend(),
                device=str(dev[0]), M=M_REPL, T=0.0, lr_rule="lr0·min(1,4/mult)", ladder=LADDER)
    print(f"[C2-R ladder] backend={meta['backend']} M={M_REPL}")
    out = {"meta": meta, "cells": {}}
    key = jr.PRNGKey(20260716 + 901)
    for mult in LADDER:
        lr = lr_for(mult)
        # --- F_s (H-R1/R2): стандартный prep, coupled Δγ=π/2, N∈{8,32} ---
        fs = {}
        for Ncell in (8, 32):
            key, sk = jr.split(key)
            mini = M.build_minimizer(dict(BASE, k_f=mult), lr=lr, freeze_w=False)
            prep = M.prep_dynamics(Ncell, True, M_REPL, sk)
            a, b0 = REF, REF; _, bp = M.apparatus_axes_theta(np.pi/2)
            ub, _ = relax(mini, prep, sk, a, b0, 0.0)
            up, _ = relax(mini, prep, sk, a, bp, 0.0)
            sb, tb, _ = signs(ub); sp, _, _ = signs(up)
            F = float(np.mean(sb != sp)); sF = float(np.sqrt(max(F*(1-F), 1e-9)/sb.size))
            D = abs(float(np.mean(sb > 0)) - float(np.mean(sp > 0)))
            fs[f"N{Ncell}"] = dict(F_s=F, sigma=sF, Delta=D)
            print(f"  k_f×{mult} N={Ncell}: F_s={F:.4f}±{sF:.4f} Δ={D:.4f}")
        # --- форма H-R4a: Haar prep, θ-сетка 9т + CHSH + R-диагностика (N=32) ---
        key, sk = jr.split(key)
        mini = M.build_minimizer(dict(BASE, k_f=mult), lr=lr, freeze_w=False)
        prep = M.prep_dynamics(32, True, M_REPL, sk)
        q = C.haar_quats(jr.fold_in(sk, 99), M_REPL)
        prepR = C.haar_rotate_prep(prep, q)
        c_diag = np.asarray((jax.vmap(rotmat)(q) @ REF) @ REF)   # (R·ê)·ê per replica
        Eth = {}
        s0_arr = None
        for th in THETAS9:
            u, _ = relax(mini, prepR, sk, axis(0.0), axis(th), T_MID)
            s, t, degen = M.classify_batch(u, axis(0.0), axis(th))
            s, t = np.asarray(s), np.asarray(t)
            E = float(np.mean(s*t))
            Eth[f"{th:.4f}"] = dict(theta=float(th), E=E, sigma=float(np.sqrt(max(1-E*E,1e-9)/s.size)),
                                    degen=float(np.mean(np.asarray(degen))))
            if abs(th) < 1e-9:
                s0_arr = s.tolist()          # per-replica s при θ=0 (для H-R6)
        # CHSH
        def Ecorr(a, b):
            u, _ = relax(mini, prepR, sk, a, b, T_MID); s, t, _ = M.classify_batch(u, a, b)
            s, t = np.asarray(s), np.asarray(t); E = float(np.mean(s*t))
            return E, float(np.sqrt(max(1-E*E,1e-9)/s.size))
        e00, x00 = Eth[f"{np.pi/4:.4f}"]["E"], Eth[f"{np.pi/4:.4f}"]["sigma"]
        e01, x01 = Eth[f"{3*np.pi/4:.4f}"]["E"], Eth[f"{3*np.pi/4:.4f}"]["sigma"]
        e10, x10 = Ecorr(axis(np.pi/2), axis(np.pi/4))
        e11, x11 = Ecorr(axis(np.pi/2), axis(3*np.pi/4))
        S = e00 - e01 + e10 + e11; sS = float(np.sqrt(x00**2+x01**2+x10**2+x11**2))
        print(f"  k_f×{mult} форма: E(0)={Eth['0.0000']['E']:+.3f} |S|={abs(S):.4f} (θ-сетка+CHSH сняты)")
        out["cells"][f"kf{mult}"] = dict(kf=mult, lr=lr, F_s=fs, theta=Eth,
            chsh=dict(S=float(S), sigma=sS), response=dict(c=c_diag.tolist(), s_theta0=s0_arr))
    with open(os.path.join(RES, "C2R_ladder_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2R_ladder_raw.json")


def main():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash>.")
    commit = sys.argv[sys.argv.index("--prereg-commit")+1]
    stage = sys.argv[sys.argv.index("--stage")+1] if "--stage" in sys.argv else "r0"
    if stage == "r0":
        stage_r0(commit)
    else:
        stage_ladder(commit)


if __name__ == "__main__":
    main()
