"""C2-J-anneal фоллоу-ап: θ-сетка+CHSH+Δ на top-A ячейке (Q4 безусловно).
Ячейка: k_c×256, N=8, slow×1, M=1200 (A=0.260). Haar-изотропизация + отжиг.
per-replica R + проекции пишутся (чек-лист).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, jax.random as jr, numpy as np
import measurement as M
from ribbon_sim.frames import rotmat
import coupling as C

HERE = os.path.dirname(__file__); RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")
BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
DT, T_HOT, BASE_PHYS, REF = 5e-3/256, 2.0, 2.4, M.REF_AXIS
KC, N, MR, SLOW = 256.0, 8, 1200, 1
THETAS9 = [i*np.pi/8 for i in range(9)]


def axis(t): return jnp.array([jnp.sin(t), 0.0, jnp.cos(t)])


def anneal(mini, prep, a, b):
    ramp = int(round(BASE_PHYS*SLOW/DT)); fin = max(1, ramp//10)
    sched = jnp.concatenate([jnp.linspace(T_HOT, 0.0, ramp), jnp.zeros(fin)])
    _, u, _ = mini["run"](jr.PRNGKey(1), prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u


def Ecorr(u, a, b):
    s, t, degen = M.classify_batch(u, a, b); s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    E = float(np.mean(s*t)); return E, float(np.sqrt(max(1-E*E, 1e-9)/s.size)), float(np.mean(degen))


def main():
    if "--prereg-commit" not in sys.argv: sys.exit("STAGED")
    commit = sys.argv[sys.argv.index("--prereg-commit")+1]
    params = dict(BASE, k_c=BASE["k_c"]*KC); mini = M.build_minimizer(params, lr=DT, freeze_w=False)
    prep = M.prep_dynamics(N, True, MR, jr.PRNGKey(20260717+999))
    q = C.haar_quats(jr.PRNGKey(31), MR); prepR = C.haar_rotate_prep(prep, q)
    print(f"[C2-J topA] k_c×{KC:.0f} N={N} slow×{SLOW} — θ-сетка+CHSH (отжиг+Haar)")
    Eth = {}
    for th in THETAS9:
        u = anneal(mini, prepR, axis(0.0), axis(th))
        E, sig, deg = Ecorr(u, axis(0.0), axis(th))
        Eth[f"{th:.4f}"] = dict(theta=float(th), E=E, sigma=sig, degen=deg)
        print(f"  θ={th:.3f}: E={E:+.4f}±{sig:.4f} degen={deg:.3f}")
    e00 = Eth[f"{np.pi/4:.4f}"]["E"]; e01 = Eth[f"{3*np.pi/4:.4f}"]["E"]
    x00 = Eth[f"{np.pi/4:.4f}"]["sigma"]; x01 = Eth[f"{3*np.pi/4:.4f}"]["sigma"]
    u10 = anneal(mini, prepR, axis(np.pi/2), axis(np.pi/4)); e10, x10, _ = Ecorr(u10, axis(np.pi/2), axis(np.pi/4))
    u11 = anneal(mini, prepR, axis(np.pi/2), axis(3*np.pi/4)); e11, x11, _ = Ecorr(u11, axis(np.pi/2), axis(3*np.pi/4))
    S = e00-e01+e10+e11; sS = float(np.sqrt(x00**2+x01**2+x10**2+x11**2))
    print(f"  |S|={abs(S):.4f}±{sS:.4f}  {'S≤2 ✓' if abs(S)<=2+sS else 'S>2 — РЕЛЬС/АУДИТ'}")
    out = dict(meta=dict(script="c2j_topA", prereg_commit=commit, cell="k_c×256 N=8 slow×1"),
               theta=Eth, chsh=dict(S=float(S), sigma=sS),
               response=dict(c=np.asarray((jax.vmap(rotmat)(q)@REF)@REF).tolist()))
    json.dump(out, open(os.path.join(RES, "C2Janneal_topA.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2Janneal_topA.json")


if __name__ == "__main__":
    main()
