#!/usr/bin/env python
"""Диагностика: истинное равновесие R1 при длительной релаксации.

Паранойя показала, что E(θ) дрейфует с числом шагов. Здесь гоним κ∈{1,10} на
coarse θ до больших steps и смотрим, к какой форме сходится E(θ) (гипотеза:
ферро-ступень при 90° для больших κ) и стабилизируется ли она.
Результат — results/R1/convergence_study.md. Пре-регистрацию не трогаем.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import numpy as np

from ribbon_sim import analysis
from ribbon_sim.dynamics import branch_counts, build_relaxer, classify
from ribbon_sim.experiment import setting_vectors
from ribbon_sim.frames import haar_quaternions

COARSE = np.radians(np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]))
STEP_LIST = (30000, 100000)


def cell_for(kappa, steps, N=64, B=8192):
    k_c = 1.0
    k_e = kappa * (N - 1) * k_c
    return {"N": N, "B": B, "k_e": k_e, "k_c": k_c, "spinor": False,
            "elastic": "geodesic", "T0": 0.0, "decay": 1.0, "steps": steps,
            "lr": 0.5 / (k_e + k_c)}


def sweep(cell, thetas, seed=0):
    r = build_relaxer(cell)
    counts = np.zeros((len(thetas), 4), dtype=np.int64)
    max_frac = 0.0
    for ti, th in enumerate(thetas):
        base = jax.random.PRNGKey(seed)
        k_init, k_noise = jax.random.split(jax.random.fold_in(base, ti))
        q0 = haar_quaternions(k_init, (cell["B"], cell["N"]))
        a, b = setting_vectors(th)
        qf, _ = r["run"](k_noise, q0, a, b)
        s, t = classify(qf, a, b)
        counts[ti] = np.asarray(branch_counts(s, t))
        frac = float(np.mean(np.asarray(r["probe"](qf, a, b)) > 1e-6))
        max_frac = max(max_frac, frac)
    return counts, max_frac


def main():
    L = []; A = L.append
    A("# R1 — исследование сходимости (диагностика)\n")
    A(f"GPU: {jax.devices()[0].device_kind}. Гоним κ до больших steps, ищем истинное "
      "равновесие E(θ). Ступень при 90° ⇒ ферро-PR-box (p→∞); плавная кривая ⇒ конечный p.\n")
    t0 = time.time()

    step_func = np.where(COARSE < np.pi / 2, 1.0, np.where(COARSE > np.pi / 2, -1.0, 0.0))

    for kappa in (1.0, 10.0):
        A(f"## κ = {kappa:g}\n")
        A("| θ° | " + " | ".join(f"E@{s}" for s in STEP_LIST) + " | ступень |")
        A("|---|" + "---|" * (len(STEP_LIST) + 1))
        Es = {}
        fracs = {}
        for steps in STEP_LIST:
            counts, mf = sweep(cell_for(kappa, steps), COARSE)
            Es[steps] = analysis.E_from_counts(counts)
            fracs[steps] = mf
        for i, th in enumerate(np.degrees(COARSE)):
            row = " | ".join(f"{Es[s][i]:+.4f}" for s in STEP_LIST)
            A(f"| {th:.0f} | {row} | {step_func[i]:+.0f} |")
        big = STEP_LIST[-1]
        drift = float(np.max(np.abs(Es[big] - Es[STEP_LIST[0]])))
        dist_step = float(np.max(np.abs(Es[big] - step_func)))
        # фит формы на самом длинном прогоне
        counts_big, _ = sweep(cell_for(kappa, big), COARSE)
        p_hat, _ = analysis.fit_chord_p(analysis.singlet_counts(counts_big), COARSE)
        sgn = -1 if analysis.E_from_counts(counts_big)[0] > 0 else 1
        h = analysis.harmonics(COARSE, Es[big])
        A(f"\n- дрейф E между {STEP_LIST[0]} и {big} шагами: **{drift:.4f}** "
          f"(доля |ΔE|/шаг>1e-6: {fracs[STEP_LIST[0]]:.3f} → {fracs[big]:.3f}).")
        A(f"- расстояние E@{big} до ступени (±1 c переходом 90°): **{dist_step:.4f}**.")
        A(f"- фит формы @ {big}: p̂={p_hat:.3f}, знак={'ферро' if sgn<0 else 'антиферро'}; "
          f"гармоники A1={h['A1']:+.3f}, A3={h['A3']:+.3f}.\n")

    A(f"---\nВремя: {time.time()-t0:.1f} с.\n")
    out = ROOT / "results" / "R1" / "convergence_study.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"conv_study → {out}")


if __name__ == "__main__":
    main()
