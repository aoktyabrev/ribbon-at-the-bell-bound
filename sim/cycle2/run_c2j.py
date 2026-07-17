"""C2-J — лестница жёсткости ЦЕПИ k_c (prereg). STAGED.
--stage j0     : J0-гейт (якорь A(k_f×1,k_c×1)≈0.369 + rej-лестница). Быстро.
--stage ladder : k_c×{4,16,64,256}, N∈{8,32}, k_f×1: F_s/F_t + форма (θ×9+CHSH)
                 + per-replica R + C_ee + пер-репличные |proj| при θ=0 (degen-гигиена).
lr-правило: lr = lr0·min(1, C/k_c_mult), C=4 (Q-аналог C2-R). T=0, M=4800, GPU fp64.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2j.py --stage {j0|ladder} --prereg-commit <hash>
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
LR0, T_MID, C_LR = 5e-3, 0.05, 4.0
M_REPL = 8 if SMOKE else 4800
STEPS = 12 if SMOKE else 4000
REF = M.REF_AXIS
KC_LADDER = [4.0, 16.0, 64.0, 256.0]
THETAS9 = [i*np.pi/8 for i in range(9)]
A_ANCHOR, S_SEED = 0.369, 0.024


def lr_for(kc_mult):
    return LR0 * min(1.0, C_LR/kc_mult)


def axis(theta):
    return jnp.array([jnp.sin(theta), 0.0, jnp.cos(theta)])


def params_kc(mult):
    return dict(BASE, k_c=BASE["k_c"]*mult)


def relax(mini, prep, key, a, b, T):
    sched = jnp.concatenate([jnp.full((STEPS,), T), jnp.zeros((STEPS//2,))])
    _, u, rej = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u, float(np.asarray(rej).sum())


def measure_A(mult, key):
    mini = M.build_minimizer(params_kc(mult), lr=lr_for(mult), freeze_w=False)
    m = 1200 if not SMOKE else 8
    prep = M.prep_dynamics(32, True, m, key)
    a, b = M.apparatus_axes_theta(0.0)
    u, rej = relax(mini, prep, key, a, b, T_MID)
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen; nv = int(keep.sum())
    E = float(np.mean((s*t)[keep])) if nv else 0.0
    return abs(E), float(np.sqrt(max(1-E*E,1e-9)/max(nv,1))), rej/(m*(STEPS+STEPS//2))


def stage_j0(commit):
    print("[C2-J J0] якорь + rej-лестница")
    key = jr.PRNGKey(20260717 + 10)
    A1, s1, _ = measure_A(1.0, key)
    scomb = float(np.sqrt(s1**2 + S_SEED**2))
    anchor_ok = abs(A1 - A_ANCHOR) <= 3*scomb
    print(f"  якорь A(k_c×1)={A1:.4f}±{s1:.4f} vs {A_ANCHOR} ⇒ {abs(A1-A_ANCHOR)/scomb:.2f}σ_comb "
          f"[{'OK' if anchor_ok else 'KILL'}]")
    rej_ok = True; tab = {}
    for mult in KC_LADDER:
        key, sk = jr.split(key)
        mini = M.build_minimizer(params_kc(mult), lr=lr_for(mult), freeze_w=False)
        m = 1200 if not SMOKE else 8
        prep = M.prep_dynamics(32, True, m, sk)
        a, b = M.apparatus_axes_theta(0.0)
        _, rej = relax(mini, prep, sk, a, b, 0.0)
        rps = rej/(m*(STEPS+STEPS//2)); ok = rps < 1e-3; rej_ok = rej_ok and ok
        tab[f"kc{mult}"] = rps
        print(f"  k_c×{mult} lr={lr_for(mult):.2e}: rej_per_step={rps:.2e} [{'OK' if ok else 'KILL'}]")
    passed = anchor_ok and rej_ok
    out = dict(meta=dict(script="run_c2j_j0", prereg_commit=commit, C_lr=C_LR),
               anchor=dict(A=A1, sigma=s1, ref=A_ANCHOR, ok=bool(anchor_ok)),
               rej=tab, J0_pass=bool(passed))
    json.dump(out, open(os.path.join(RES, "C2J_J0.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2J_J0.json  J0-ГЕЙТ: {'PASS' if passed else 'KILL — стоп'}")
    if not passed:
        sys.exit(1)


def signs_proj(u, a, b):
    """s,t,degen + |proj_A|,|proj_B| (для degen-гигиены)."""
    s, t, pa, pb = jax.vmap(M.classify_raw, in_axes=(0, None, None))(u, a, b)
    return np.asarray(s), np.asarray(t), np.asarray(pa), np.asarray(pb)


def stage_ladder(commit):
    dev = jax.devices()
    meta = dict(script="run_c2j_ladder", prereg_commit=commit, backend=jax.default_backend(),
                device=str(dev[0]), M=M_REPL, T=0.0, kf=1.0, ladder=KC_LADDER, C_lr=C_LR)
    print(f"[C2-J ladder] backend={meta['backend']} M={M_REPL}")
    out = {"meta": meta, "cells": {}}
    key = jr.PRNGKey(20260717 + 11)
    for mult in KC_LADDER:
        lr = lr_for(mult); pr = params_kc(mult)
        # F_s/F_t (H-J1): стандартный prep, coupled Δγ=π/2, N∈{8,32}
        fs = {}
        for Ncell in (8, 32):
            key, sk = jr.split(key)
            mini = M.build_minimizer(pr, lr=lr, freeze_w=False)
            prep = M.prep_dynamics(Ncell, True, M_REPL, sk)
            a, b0 = REF, REF; _, bp = M.apparatus_axes_theta(np.pi/2)
            ub, _ = relax(mini, prep, sk, a, b0, 0.0); up, _ = relax(mini, prep, sk, a, bp, 0.0)
            sb, tb, _, _ = signs_proj(ub, REF, REF); sp, _, _, _ = signs_proj(up, REF, REF)
            _, tbp, _, _ = signs_proj(up, REF, bp)   # F_t: дальний своей осью
            F = float(np.mean(sb != sp)); sF = float(np.sqrt(max(F*(1-F),1e-9)/sb.size))
            _, tb_own, _, _ = signs_proj(ub, REF, b0)
            Ft = float(np.mean(tb_own != tbp))
            D = abs(float(np.mean(sb > 0)) - float(np.mean(sp > 0)))
            fs[f"N{Ncell}"] = dict(F_s=F, sigma=sF, F_t=Ft, Delta=D)
            print(f"  k_c×{mult} N={Ncell}: F_s={F:.4f}±{sF:.4f} F_t={Ft:.4f} Δ={D:.4f}")
        # форма: Haar prep, θ-сетка + CHSH + R + C_ee + proj при θ=0
        key, sk = jr.split(key)
        mini = M.build_minimizer(pr, lr=lr, freeze_w=False)
        prep = M.prep_dynamics(32, True, M_REPL, sk)
        q = C.haar_quats(jr.fold_in(sk, 99), M_REPL); prepR = C.haar_rotate_prep(prep, q)
        c_diag = np.asarray((jax.vmap(rotmat)(q) @ REF) @ REF)
        Eth = {}; s0 = t0 = pa0 = pb0 = None; cee = None
        for th in THETAS9:
            u, _ = relax(mini, prepR, sk, axis(0.0), axis(th), T_MID)
            s, t, pa, pb = signs_proj(u, axis(0.0), axis(th))
            E = float(np.mean(s*t))
            Eth[f"{th:.4f}"] = dict(theta=float(th), E=E, sigma=float(np.sqrt(max(1-E*E,1e-9)/s.size)),
                                    degen=float(np.mean((pa<0.2)|(pb<0.2))))
            if abs(th) < 1e-9:
                s0 = s.tolist(); t0 = t.tolist(); pa0 = pa.tolist(); pb0 = pb.tolist()
                nA = jax.vmap(lambda uu: M.end_axis(uu[0]))(u); nB = jax.vmap(lambda uu: M.end_axis(uu[-1]))(u)
                cee = float(np.mean(np.sum(np.asarray(nA)*np.asarray(nB), -1)))
        def Ecorr(a, b):
            u, _ = relax(mini, prepR, sk, a, b, T_MID); s, t, _, _ = signs_proj(u, a, b)
            E = float(np.mean(s*t)); return E, float(np.sqrt(max(1-E*E,1e-9)/s.size))
        e00 = Eth[f"{np.pi/4:.4f}"]["E"]; x00 = Eth[f"{np.pi/4:.4f}"]["sigma"]
        e01 = Eth[f"{3*np.pi/4:.4f}"]["E"]; x01 = Eth[f"{3*np.pi/4:.4f}"]["sigma"]
        e10, x10 = Ecorr(axis(np.pi/2), axis(np.pi/4)); e11, x11 = Ecorr(axis(np.pi/2), axis(3*np.pi/4))
        S = e00-e01+e10+e11; sS = float(np.sqrt(x00**2+x01**2+x10**2+x11**2))
        print(f"  k_c×{mult} форма: E(0)={Eth['0.0000']['E']:+.3f} |S|={abs(S):.4f} C_ee={cee:+.3f}")
        # рельс {F_s=0, S>2}
        rail = (fs["N32"]["F_s"] < 3*fs["N32"]["sigma"]) and (abs(S) > 2 + sS)
        if rail:
            print(f"    ⚠⚠ РЕЛЬС: F_s≈0 И |S|={abs(S):.3f}>2 в k_c×{mult} — БЕЗУСЛОВНЫЙ СТОП-АУДИТ")
        out["cells"][f"kc{mult}"] = dict(kc=mult, lr=lr, F_s=fs, theta=Eth, C_ee=cee,
            chsh=dict(S=float(S), sigma=sS),
            response=dict(c=c_diag.tolist(), s_theta0=s0, t_theta0=t0, proj_A0=pa0, proj_B0=pb0),
            rail_violation=bool(rail))
        if rail and not SMOKE:
            with open(os.path.join(RES, "C2J_ladder_raw.json"), "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            sys.exit(f"РЕЛЬС сорван на k_c×{mult}: стоп-аудит, сырьё сохранено.")
    with open(os.path.join(RES, "C2J_ladder_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2J_ladder_raw.json")


def main():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash>.")
    commit = sys.argv[sys.argv.index("--prereg-commit")+1]
    stage = sys.argv[sys.argv.index("--stage")+1] if "--stage" in sys.argv else "j0"
    stage_j0(commit) if stage == "j0" else stage_ladder(commit)


if __name__ == "__main__":
    main()
