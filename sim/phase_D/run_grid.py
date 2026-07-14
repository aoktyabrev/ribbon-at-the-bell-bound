"""D2-slim. Сетка сырых измерений E_raw(θ) (см. D2_prereg.md, аддендум §1–§3).

ТОЛЬКО СЫРЬЁ: совместные таблицы (s,t) по AX, E_raw=⟨s·t⟩, σ. НИКАКИХ флипов/фитов/
сверок (порядок анализа §4: сырьё → коммит → анализ). Результат → D2_raw_tables.json.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_grid.py [--smoke]
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
import readout as R

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR = 5e-3
T_MID = 0.05
T_LOW = 0.02
THETAS = [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi]
THETA_NAMES = ["0", "pi/8", "pi/4", "3pi/8", "pi/2", "5pi/8", "3pi/4", "7pi/8", "pi"]
if SMOKE:
    NS = [16]
    M_REPL = 80
    STEPS = 1500
    THETAS = [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    THETA_NAMES = ["0", "pi/4", "pi/2", "3pi/4", "pi"]
else:
    NS = [16, 32, 48]
    M_REPL = 300
    STEPS = 4000


def measure_point(N, sector_odd, theta, T, key, candidate="AX"):
    """Одна точка: M реплик, релаксация+доохлаждение, таблица (s,t) по кандидату."""
    a, b = M.apparatus_axes_theta(theta)
    mini = M.build_minimizer(PARAMS, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, sector_odd, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, rej1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b,
                             jnp.full((STEPS,), T))
    x, u, rej2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b,
                             jnp.full((STEPS // 2,), 0.0))
    rej = float(np.asarray(rej1).sum() + np.asarray(rej2).sum())
    if candidate == "AX":
        s, t, degen = M.classify_batch(u, a, b)
        s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    else:
        Lb, Rb = jax.vmap(R.CANDIDATES[candidate], in_axes=(0, 0, None, None))(x, u, a, b)
        s, t = np.asarray(Lb), np.asarray(Rb)
        _, _, degen = M.classify_batch(u, a, b)
        degen = np.asarray(degen)
    keep = ~degen
    sv, tv = s[keep], t[keep]
    tab = dict(pp=int(((sv > 0) & (tv > 0)).sum()),
               pm=int(((sv > 0) & (tv < 0)).sum()),
               mp=int(((sv < 0) & (tv > 0)).sum()),
               mm=int(((sv < 0) & (tv < 0)).sum()),
               degen=int(degen.sum()))
    n_valid = tab["pp"] + tab["pm"] + tab["mp"] + tab["mm"]
    E = (tab["pp"] + tab["mm"] - tab["pm"] - tab["mp"]) / max(n_valid, 1)
    sigma = float(np.sqrt(max(1.0 - E*E, 1e-9) / max(n_valid, 1)))
    return dict(table=tab, E_raw=float(E), sigma=sigma, n_valid=n_valid,
                rej_per_step=rej / (M_REPL * (STEPS + STEPS // 2)))


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260714)
    print(f"[grid {tag}] N={NS} M={M_REPL} θ={THETA_NAMES} T_mid={T_MID}")

    data = {"meta": dict(tag=tag, NS=NS, M=M_REPL, thetas=THETA_NAMES,
                         theta_vals=[float(t) for t in THETAS],
                         T_mid=T_MID, T_low=T_LOW, params=PARAMS,
                         commit_base="81cc7da", note="СЫРЫЕ таблицы AX; флип/фиты — ПОСЛЕ коммита"),
            "AX": {}, "control_AC": {}, "T_low": {}, "mirror": {}}

    # --- основная сетка AX ---
    for N in NS:
        for sector_odd in [False, True]:
            sname = "odd" if sector_odd else "even"
            for ti, theta in enumerate(THETAS):
                key, sk = jr.split(key)
                r = measure_point(N, sector_odd, theta, T_MID, sk)
                cell = f"N{N}|{sname}|{THETA_NAMES[ti]}"
                data["AX"][cell] = r
                print(f"  {cell:20s}: E_raw={r['E_raw']:+.3f}±{r['sigma']:.3f} "
                      f"tab={r['table']} rej={r['rej_per_step']:.1e}")

    # --- контроль A/C при θ∈{0,π/2,π} ---
    ctrl_thetas = [(0.0, "0"), (np.pi/2, "pi/2"), (np.pi, "pi")]
    for N in ([NS[-1]] if not SMOKE else NS):
        for sector_odd in [False, True]:
            sname = "odd" if sector_odd else "even"
            for theta, thn in ctrl_thetas:
                for cand in ["A", "C"]:
                    key, sk = jr.split(key)
                    r = measure_point(N, sector_odd, theta, T_MID, sk, candidate=cand)
                    data["control_AC"][f"N{N}|{sname}|{thn}|{cand}"] = r

    # --- T_low контроль при θ=π/4 ---
    for N in NS:
        for sector_odd in [False, True]:
            sname = "odd" if sector_odd else "even"
            key, sk = jr.split(key)
            r = measure_point(N, sector_odd, np.pi/4, T_LOW, sk)
            data["T_low"][f"N{N}|{sname}|pi/4"] = r

    # --- зеркальный контроль (a,b)→(−a,−b) при θ∈{0,π/4,π/2}, нечёт, N-макс ---
    Nm = NS[-1]
    for theta, thn in [(0.0, "0"), (np.pi/4, "pi/4"), (np.pi/2, "pi/2")]:
        key, sk = jr.split(key)
        a, b = M.apparatus_axes_theta(theta)
        mini = M.build_minimizer(PARAMS, lr=LR, freeze_w=False)
        p = M.prep_dynamics(Nm, True, M_REPL, sk)
        k1, k2 = jr.split(sk)
        x, u, _ = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
        x, u, _ = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS//2,), 0.0))
        s, t, degen = M.classify_batch(u, a, b)
        sm, tm, _ = M.classify_batch(u, -a, -b)
        s, t, sm, tm, degen = [np.asarray(z) for z in (s, t, sm, tm, degen)]
        keep = ~degen
        E = float(np.mean((s*t)[keep])); Em = float(np.mean((sm*tm)[keep]))
        data["mirror"][f"N{Nm}|odd|{thn}"] = dict(E_raw=E, E_mirror=Em, match=bool(abs(E-Em) < 0.05))
        print(f"  mirror N{Nm} odd θ={thn}: E={E:+.3f} E_mirror={Em:+.3f}")

    with open(os.path.join(RES, "D2_raw_tables.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  СЫРЫЕ таблицы → {RES}/D2_raw_tables.json (далее: КОММИТ, потом анализ)")


if __name__ == "__main__":
    main()
