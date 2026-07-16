"""C2-TM этап 2 (prereg 97b9622, predictions 99ad201). STAGED.
Новые точки A(k_f) при k_f∈{0.25, 8} (+ 6.0 по бюджету), протокол S1-runs R1
(нечёт, θ=0, N=32, M=1200, T_mid=0.05 + доохлаждение) — идентичен a9cef7b.
Бэкенд GPU/fp64 (prereg; V0 подтвердил воспроизводимость эталона).

ВНИМАНИЕ (записать в отчёт): замороженные 4 точки считались на CPU; эти —
на GPU. V0 показал совпадение при k_f×4 (0.13σ), но расхождение при k_f×1
(2.51σ). Возможный CPU/GPU-конфаунд в объединённом фите — флаг, не правка.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2tm_stage2.py --prereg-commit <hash>
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

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR, T_MID, N, M_REPL, STEPS = 5e-3, 0.05, 32, 1200, 4000
KFS = [0.25, 6.0, 8.0]


def measure(mult, key):
    """Ячейка R1 (нечёт, θ=0): релаксация T_mid + доохлаждение, A=|⟨s·t⟩| по AX + зеркало."""
    params = dict(BASE, k_f=BASE["k_f"] * mult)
    a, b = M.apparatus_axes_theta(0.0)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    s, t, degen = M.classify_batch(u, a, b)
    sm, tm, _ = M.classify_batch(u, -a, -b)
    s, t, sm, tm, degen = [np.asarray(z) for z in (s, t, sm, tm, degen)]
    keep = ~degen; nv = int(keep.sum())
    E = float(np.mean((s * t)[keep])) if nv else 0.0
    Emir = float(np.mean((sm * tm)[keep])) if nv else 0.0
    rej = float(np.asarray(r1).sum() + np.asarray(r2).sum())
    return dict(k_f=mult, E_raw=E, A=abs(E), sigma=float(np.sqrt(max(1 - E*E, 1e-9)/max(nv, 1))),
                E_mirror=Emir, P_aligned=float(np.mean(((s*t) > 0)[keep])) if nv else 0.0,
                n_valid=nv, degen=int(degen.sum()), rej_per_step=rej/(M_REPL*(STEPS+STEPS//2)))


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash>.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2tm_stage2", prereg_commit=commit, predictions_commit="99ad201",
                backend=jax.default_backend(), jax_version=jax.__version__, device=str(dev[0]),
                x64=jax.config.jax_enable_x64, N=N, M=M_REPL, T_mid=T_MID, protocol="a9cef7b S1-R1",
                note="frozen 4 pts=CPU; these=GPU (V0-confound flag)")
    print(f"[C2-TM stage2] backend={meta['backend']} device={meta['device']} k_f={KFS}")
    out = {"meta": meta, "points": {}}
    key = jr.PRNGKey(20260716 + 300)
    for mult in KFS:
        key, sk = jr.split(key)
        r = measure(mult, sk)
        out["points"][f"kf{mult}"] = r
        print(f"  k_f={mult}: A={r['A']:.4f}±{r['sigma']:.4f} P_al={r['P_aligned']:.3f} "
              f"rej={r['rej_per_step']:.1e} degen={r['degen']}")
    with open(os.path.join(RES, "C2TM_stage2_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2TM_stage2_raw.json (далее: КОММИТ, потом анализ против KILL)")


if __name__ == "__main__":
    main()
