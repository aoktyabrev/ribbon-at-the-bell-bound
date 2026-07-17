"""C2-J-anneal кампания (prereg 2cc996d + budget-addendum). STAGED.
--stage regate   : F0-ре-гейт (δ-лестница) на k_c×64×slow×64, N=32, M=1200.
--stage campaign : ядро — k_c×{16,64}×N∈{8,32}×slow×{1,4,16,64} M=4800 (полн.
                   наблюдаемые); k_c×256 slow×{1,4} M=1200 (A/E_res/C_ee/E_std).
Отжиг: T(t)=T_hot·(1−t/t_ramp) линейно + T=0 доводка (10%). dt арбитражный.
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
DT0, T_HOT, REF = 5e-3, 2.0, M.REF_AXIS
BASE_PHYS = 0.02 if SMOKE else 2.4
DT_CELL = {16: DT0/16, 64: DT0/16, 256: DT0/256}
THETAS9 = [i*np.pi/8 for i in range(9)]


def dt_of(mult):
    return DT_CELL[int(mult)]


def anneal_sched(mult, slow):
    dt = dt_of(mult)
    ramp = int(round(BASE_PHYS*slow/dt)); fin = max(1, ramp//10)
    return jnp.concatenate([jnp.linspace(T_HOT, 0.0, ramp), jnp.zeros(fin)]), ramp+fin, dt


def anneal(params, prep, mult, slow, a, b, key=jr.PRNGKey(1)):
    sched, nsteps, dt = anneal_sched(mult, slow)
    mini = M.build_minimizer(params, lr=dt, freeze_w=False)
    x, u, rej = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return x, u, float(np.asarray(rej).sum())/(prep["x0"].shape[0]*nsteps), nsteps


def obs_A(x, u, a, b, params):
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen; nv = int(keep.sum())
    A = abs(float(np.mean((s*t)[keep]))) if nv else 0.0
    nA = jax.vmap(lambda uu: M.end_axis(uu[0]))(u); nB = jax.vmap(lambda uu: M.end_axis(uu[-1]))(u)
    cee = float(np.mean(np.sum(np.asarray(nA)*np.asarray(nB), -1)))
    Ef = np.asarray(jax.vmap(lambda xx, uu: M.e_meas(xx, uu, a, b, params))(x, u))
    return dict(A=A, sigma=float(np.sqrt(max(1-A*A,1e-9)/max(nv,1))), C_ee=cee,
                E_res=float(Ef.mean()), E_std=float(Ef.std()), degen=float(np.mean(degen)))


def _req():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: --prereg-commit <hash>.")
    return sys.argv[sys.argv.index("--prereg-commit")+1]


def stage_regate(commit):
    print("[C2-J regate] F0 δ-лестница на k_c×64×slow×64, N=32, M=1200")
    mult, slow, N, Mr = 64.0, 64, 32, 1200
    params = dict(BASE, k_c=BASE["k_c"]*mult)
    prep = M.prep_dynamics(N, True, Mr, jr.PRNGKey(20260717+70))
    a, b0 = REF, REF
    deltas = [1e-6, 1e-4, 1e-2, np.pi/8, np.pi/2]
    # нуль-тест γ=0
    xb, ub, rej, ns = anneal(params, prep, mult, slow, a, b0)
    x2, u2, _, _ = anneal(params, prep, mult, slow, a, b0)
    null = float(np.max(np.abs(np.asarray(ub)-np.asarray(u2))))
    print(f"  нуль-тест γ=0: max|Δu|={null:.1e} ({ns} шаг/отжиг); rej={rej:.1e}")
    sb, _, _ = obs_signs(ub, a, b0)
    Ftab = {}
    for d in deltas:
        _, bp = M.apparatus_axes_theta(d)
        xp, up, _, _ = anneal(params, prep, mult, slow, a, bp)
        sp, _, _ = obs_signs(up, a, REF)
        sb2, _, _ = obs_signs(ub, a, REF)
        F = float(np.mean(sb2 != sp)); sF = float(np.sqrt(max(F*(1-F),1e-9)/sb2.size))
        Ftab[f"{d:.3e}"] = dict(F=F, sigma=sF)
        print(f"  δ={d:.1e}: F={F:.4f}±{sF:.4f}")
    F1e6 = Ftab[f"{1e-6:.3e}"]["F"]
    passed = (F1e6 == 0.0)   # ветка (i) проход (F0 стандарт)
    out = dict(meta=dict(script="c2j_regate", prereg_commit=commit, cell="k_c×64 slow×64"),
               null_maxdiff=null, F=Ftab, gate_pass_i=bool(passed))
    json.dump(out, open(os.path.join(RES, "C2Janneal_regate.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → регейт: F0 ветка (i) {'ПРОХОД (F_s легальна k_c≤64)' if passed else 'НЕ проход — F_s дистрибутивно'}")


def obs_signs(u, a, b):
    s, t, degen = M.classify_batch(u, a, b)
    return np.asarray(s), np.asarray(t), np.asarray(degen)


def stage_campaign(commit):
    dev = jax.devices()
    out = {"meta": dict(script="c2j_campaign", prereg_commit=commit, backend=jax.default_backend(),
                        device=str(dev[0]), T_hot=T_HOT, base_phys=BASE_PHYS), "cells": {}}
    print(f"[C2-J campaign] backend={out['meta']['backend']} T_hot={T_HOT}")
    a, b = M.apparatus_axes_theta(np.pi)          # b=−a
    plan = []
    for mult in (16.0, 64.0):
        for N in (8, 32):
            for slow in (1, 4, 16, 64):
                plan.append((mult, N, slow, 4800))
    for N in (8, 32):
        for slow in (1, 4):
            plan.append((256.0, N, slow, 1200))
    for mult, N, slow, Mr in plan:
        if SMOKE:
            Mr = 8
        params = dict(BASE, k_c=BASE["k_c"]*mult)
        prep = M.prep_dynamics(N, True, Mr, jr.PRNGKey(20260717+int(mult)*10+N+slow))
        x, u, rej, ns = anneal(params, prep, mult, slow, a, b)
        o = obs_A(x, u, a, b, params); o.update(rej_per_step=rej, steps=ns)
        out["cells"][f"kc{mult}|N{N}|slow{slow}"] = dict(kc=mult, N=N, slow=slow, M=Mr, **o)
        print(f"  k_c×{mult:.0f} N={N} slow×{slow}: A={o['A']:.4f}±{o['sigma']:.4f} "
              f"C_ee={o['C_ee']:+.3f} E_std={o['E_std']:.2f} rej={rej:.1e} ({ns} шаг)")
        with open(os.path.join(RES, "C2Janneal_campaign.json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)   # инкрементально
    print(f"  СЫРЬЁ → {RES}/C2Janneal_campaign.json")


def main():
    commit = _req()
    stage = sys.argv[sys.argv.index("--stage")+1] if "--stage" in sys.argv else "campaign"
    stage_regate(commit) if stage == "regate" else stage_campaign(commit)


if __name__ == "__main__":
    main()
