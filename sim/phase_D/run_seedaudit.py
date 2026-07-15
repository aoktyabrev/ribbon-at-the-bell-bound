"""SEED-AUDIT. Межсидовая воспроизводимость фазы D (см. seedaudit_prereg.md). ТОЛЬКО СЫРЬЁ.

SA-a ядро: нечёт, N=32, k_f×1, T=0.05, M=1200, K=8 свежих базовых ключей.
SA-b край:  нечёт, N=96, k_f×1, T=0.05, M=1200, K=4 свежих базовых ключа.

measure() перенесён из run_s1.py БЕЗ изменений (параметризован только N) — сравнение
корректно лишь при тождественном протоколе. Результат → seedaudit_raw.json;
s_seed/r/фиты — ПОСЛЕ коммита сырья.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_seedaudit.py [--smoke]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

# --- база: идентична run_s1.py, не трогать (SPEC §2.2; seedaudit_prereg «База») ---
BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR = 5e-3
T_MID = 0.05
M_REPL = 150 if SMOKE else 1200
STEPS = 1500 if SMOKE else 4000

# --- свежие базовые сиды, зафиксированы в seedaudit_prereg.md ДО прогонов ---
SEEDS_CORE = [11235813, 27182818, 31415926, 16180339, 14142135, 57721566, 26457513, 69314718]
SEEDS_N96 = [22026465, 17320508, 12246467, 86602540]

N_CORE = 16 if SMOKE else 32
N_EDGE = 24 if SMOKE else 96
K_CORE = 2 if SMOKE else 8
K_EDGE = 2 if SMOKE else 4


def measure(N, sector_odd, params, T, key):
    """Точка θ=0: M реплик, релаксация T + доохлаждение, A=|⟨s·t⟩| по AX + зеркало.

    Перенесено из run_s1.py::measure без изменений (N параметризован)."""
    a, b = M.apparatus_axes_theta(0.0)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, sector_odd, M_REPL, key)
    k1, k2 = jr.split(key)
    x, u, r1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T))
    x, u, r2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    rej = float(np.asarray(r1).sum() + np.asarray(r2).sum())
    s, t, degen = M.classify_batch(u, a, b)
    sm, tm, _ = M.classify_batch(u, -a, -b)
    s, t, sm, tm, degen = [np.asarray(z) for z in (s, t, sm, tm, degen)]
    keep = ~degen
    sv, tv = s[keep], t[keep]
    n_valid = int(keep.sum())
    E = float(np.mean((sv * tv))) if n_valid else 0.0
    Emir = float(np.mean((sm * tm)[keep])) if n_valid else 0.0
    P_al = float(np.mean(((sv * tv) > 0))) if n_valid else 0.0
    sigma = float(np.sqrt(max(1.0 - E * E, 1e-9) / max(n_valid, 1)))
    return dict(E_raw=E, A=abs(E), sigma=sigma, E_mirror=Emir, P_aligned=P_al,
                n_valid=n_valid, degen=int(degen.sum()),
                rej_per_step=rej / (M_REPL * (STEPS + STEPS // 2)))


def run_block(name, N, seeds, out):
    for i, sd in enumerate(seeds):
        t0 = time.time()
        # СВЕЖИЙ базовый ключ: PRNGKey(seed) напрямую, не производный от 20260714
        r = measure(N, True, BASE, T_MID, jr.PRNGKey(sd))
        r["seed"] = sd
        out[f"seed{sd}"] = r
        print(f"  {name} [{i+1}/{len(seeds)}] seed={sd}: A={r['A']:.4f}±{r['sigma']:.4f} "
              f"P_al={r['P_aligned']:.3f} degen={r['degen']} ({time.time()-t0:.0f}s)",
              flush=True)


def main():
    tag = "smoke" if SMOKE else "full"
    t_start = time.time()
    print(f"[seed-audit {tag}] core N={N_CORE} K={K_CORE} | edge N={N_EDGE} K={K_EDGE} "
          f"M={M_REPL} θ=0 нечёт", flush=True)
    print(f"  devices: {jax.devices()}", flush=True)

    data = {"meta": dict(tag=tag, N_core=N_CORE, N_edge=N_EDGE, M=M_REPL, T_mid=T_MID,
                         base=BASE, lr=LR, steps=STEPS,
                         seeds_core=SEEDS_CORE[:K_CORE], seeds_edge=SEEDS_N96[:K_EDGE],
                         commit_base="6f94c04", prereg="seedaudit_prereg.md",
                         note="СЫРЬЁ AX θ0 нечёт; s_seed/r/фиты — ПОСЛЕ коммита"),
            "SA_a_core": {}, "SA_b_edge": {}}

    run_block("SA-a core", N_CORE, SEEDS_CORE[:K_CORE], data["SA_a_core"])
    run_block("SA-b edge", N_EDGE, SEEDS_N96[:K_EDGE], data["SA_b_edge"])

    out = os.path.join(RES, f"seedaudit_raw{'_smoke' if SMOKE else ''}.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  СЫРЬЁ → {out} ({time.time()-t_start:.0f}s всего; далее: КОММИТ, потом анализ)")


if __name__ == "__main__":
    main()
