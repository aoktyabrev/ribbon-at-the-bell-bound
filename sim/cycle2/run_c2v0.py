"""C2-V0 — валидация бэкенда GPU-fp64 (см. prereg_drafts/C2V0_prereg_DRAFT.md).
STAGED: НЕ запускать до утверждения Артемом и prereg-коммита.

Воспроизводит на GPU/fp64 две замороженные ячейки S1-runs R1 (протокол a9cef7b,
СВЕЖИЕ seed): k_f×1 (эталон A=0.418±0.026) и k_f×4 (A=0.837±0.016), M=1200.
Гейт: |A_gpu − A_frozen| ≤ 3σ_comb, σ_comb=√(σ_bin²+s_seed²), s_seed=0.024.
A здесь — УЖЕ опубликованная замороженная наблюдаемая, не prereg-величина
цикла 2 ⇒ сравнение легально (это тест инфраструктуры, не подглядывание).

Запуск (после утверждения): PYTHONPATH=src:phase_D:cycle2 \
    python cycle2/run_c2v0.py --prereg-commit <hash>
Бэкенд GPU по умолчанию (JAX_PLATFORMS не задаётся). fp64 включён.
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
S_SEED = 0.024                       # seed-audit a26f76b
FROZEN = {1.0: (0.418, 0.026), 4.0: (0.837, 0.016)}   # эталон a9cef7b (A, σ_bin)


def measure(params, key):
    """Ячейка R1 при θ=0, нечёт: релаксация T_MID + доохлаждение, A=|⟨s·t⟩| по AX."""
    a, b = M.apparatus_axes_theta(0.0)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen
    nv = int(keep.sum())
    E = float(np.mean((s * t)[keep])) if nv else 0.0
    rej = float(np.asarray(r1).sum() + np.asarray(r2).sum())
    return dict(A=abs(E), sigma_bin=float(np.sqrt(max(1 - E*E, 1e-9)/max(nv, 1))),
                n_valid=nv, degen=int(degen.sum()),
                rej_per_step=rej/(M_REPL*(STEPS+STEPS//2)))


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash> после утверждения Артемом.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2v0", prereg_commit=commit,
                backend=jax.default_backend(), jax_version=jax.__version__,
                device=str(dev[0]), x64=jax.config.jax_enable_x64,
                N=N, M=M_REPL, steps=STEPS, protocol="a9cef7b S1-R1")
    print(f"[C2-V0] backend={meta['backend']} device={meta['device']} jax={meta['jax_version']}")
    out = {"meta": meta, "cells": {}}
    key = jr.PRNGKey(20260716 + 0)
    all_ok = True
    for mult, (A_ref, sig_bin_ref) in FROZEN.items():
        key, sk = jr.split(key)
        r = measure(dict(BASE, k_f=BASE["k_f"] * mult), sk)
        sig_comb = float(np.sqrt(r["sigma_bin"]**2 + S_SEED**2))
        dev_sig = abs(r["A"] - A_ref) / sig_comb
        ok = dev_sig <= 3.0
        all_ok &= ok
        r.update(A_ref=A_ref, sigma_comb=sig_comb, dev_sigma=dev_sig, ok=bool(ok))
        out["cells"][f"kf{mult}"] = r
        print(f"  k_f×{mult}: A_gpu={r['A']:.3f}±{r['sigma_bin']:.3f} vs эталон {A_ref:.3f} "
              f"⇒ {dev_sig:.2f}σ_comb  rej/step={r['rej_per_step']:.1e}  [{'PASS' if ok else 'KILL'}]")
    out["gate_pass"] = bool(all_ok)
    with open(os.path.join(RES, "C2V0_raw.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {RES}/C2V0_raw.json  ГЕЙТ: {'PASS' if all_ok else 'KILL — стоп, аудит бэкенда'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
