#!/usr/bin/env python
"""Протокол паранойи SPEC §9.2 для R1 (запускается вручную, не из run.py).

Повод: κ=10 дал форму +cos θ (p=2 по форме, ферро-знак) ВОПРЕКИ пре-регистрации,
и ни одна ячейка не прошла энергетический критерий сходимости. Проверяем:
  1) стабильность НАБЛЮДАЕМОЙ E(θ) по числу шагов (сошлась ли классификация,
     несмотря на микро-дрейф энергии);
  2) устойчивость к удвоению N и B;
  3) знак корреляции (выравнивание концов) — не баг ли классификатора;
  4) контроль κ=0.1: восстанавливаются ли маргиналы при больших steps.
Результат — results/R1/paranoia.md. Пре-регистрацию не трогаем.
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


def run_point(cell, theta, seed):
    r = build_relaxer(cell)
    base = jax.random.PRNGKey(int(seed))
    k_init, k_noise = jax.random.split(jax.random.fold_in(base, 0))
    q0 = haar_quaternions(k_init, (cell["B"], cell["N"]))
    a, b = setting_vectors(theta)
    qf, _ = r["run"](k_noise, q0, a, b)
    s, t = classify(qf, a, b)
    cnt = np.asarray(branch_counts(s, t))
    same_frac = float(np.mean(np.asarray(s) == np.asarray(t)))
    return cnt, same_frac


def sweep(cell, thetas, seeds):
    counts = np.zeros((len(seeds), len(thetas), 4), dtype=np.int64)
    for si, seed in enumerate(seeds):
        for ti, th in enumerate(thetas):
            cnt, _ = run_point(cell, th, seed)
            counts[si, ti] = cnt
    return counts.sum(0)


def base_cell(kappa, N=64, B=8192, steps=6000, lr=None):
    k_c = 1.0
    k_e = kappa * (N - 1) * k_c
    return {"N": N, "B": B, "k_e": k_e, "k_c": k_c, "spinor": False,
            "elastic": "geodesic", "T0": 0.0, "decay": 1.0, "steps": steps,
            "lr": lr if lr is not None else 0.5 / (k_e + k_c)}


def main():
    L = []
    A = L.append
    A("# R1 — протокол паранойи (SPEC §9.2)\n")
    A(f"GPU: {jax.devices()[0].device_kind}. Повод: κ=10 → форма +cos θ (p≈2, ферро) "
      "вопреки пре-регистрации; энергетический критерий сходимости не пройден.\n")
    t_all = time.time()

    # ---- 1. Стабильность E(θ) по числу шагов (κ=10) --------------------------
    A("## 1. Сходимость НАБЛЮДАЕМОЙ: E(θ) vs число шагов (κ=10, N=64, B=8192, 1 сид)\n")
    A("| θ° | E@3000 | E@6000 | E@12000 | E@24000 |")
    A("|---|---|---|---|---|")
    Es = {}
    for steps in (3000, 6000, 12000, 24000):
        cell = base_cell(10.0, steps=steps)
        counts = sweep(cell, COARSE, [0])
        Es[steps] = analysis.E_from_counts(counts)
    for i, th in enumerate(np.degrees(COARSE)):
        A(f"| {th:.0f} | {Es[3000][i]:+.4f} | {Es[6000][i]:+.4f} | "
          f"{Es[12000][i]:+.4f} | {Es[24000][i]:+.4f} |")
    drift = float(np.max(np.abs(Es[24000] - Es[6000])))
    A(f"\nМакс |E@24000 − E@6000| = **{drift:.4f}**. "
      f"{'E(θ) СТАБИЛЬНА → наблюдаемая сошлась, микро-дрейф энергии не влияет на ветви.' if drift < 0.02 else 'E(θ) дрейфует — наблюдаемая НЕ сошлась!'}\n")

    # ---- 2. Удвоение N и B (κ=10) -------------------------------------------
    A("## 2. Устойчивость к удвоению N и B (κ=10, 2 сида, coarse θ)\n")
    ref = sweep(base_cell(10.0, N=64, B=8192, steps=6000), COARSE, [0, 1])
    big = sweep(base_cell(10.0, N=128, B=16384, steps=6000), COARSE, [0, 1])
    E_ref, E_big = analysis.E_from_counts(ref), analysis.E_from_counts(big)
    A("| θ° | E(N=64,B=8192) | E(N=128,B=16384) | ΔE |")
    A("|---|---|---|---|")
    for i, th in enumerate(np.degrees(COARSE)):
        A(f"| {th:.0f} | {E_ref[i]:+.4f} | {E_big[i]:+.4f} | {E_big[i]-E_ref[i]:+.4f} |")
    dmax = float(np.max(np.abs(E_big - E_ref)))
    A(f"\nМакс |ΔE| при удвоении N,B = **{dmax:.4f}**. "
      f"{'Форма устойчива → не конечно-размерный артефакт.' if dmax < 0.03 else 'ЗАВИСИТ от N,B — подозрительно!'}\n")
    # фиты формы на большом прогоне
    p_big, _ = analysis.fit_chord_p(analysis.singlet_counts(big), COARSE)
    sgn_big = -1 if analysis.E_from_counts(big)[0] > 0 else 1
    h_big = analysis.harmonics(COARSE, E_big)
    A(f"Фит формы (N=128,B=16384): p̂={p_big:.3f}, знак={'ферро' if sgn_big<0 else 'антиферро'}; "
      f"гармоники A1={h_big['A1']:+.3f}, A3={h_big['A3']:+.3f}.\n")

    # ---- 3. Знак корреляции — не баг ли классификатора ----------------------
    A("## 3. Проверка знака: доля лент с s==t (κ=10, N=64, B=8192)\n")
    A("Ожидание при выравнивании концов: θ=0 → s==t почти всегда; θ=180 → s==−t.\n")
    A("| θ° | доля s==t |")
    A("|---|---|")
    cell = base_cell(10.0, steps=6000)
    for th in np.radians([0.0, 90.0, 180.0]):
        _, same = run_point(cell, th, 0)
        A(f"| {np.degrees(th):.0f} | {same:.4f} |")
    A("\nЕсли θ=0 → ~1.0 и θ=180 → ~0.0, знак E(0)=+1 корректен: концы выравниваются "
      "(ферро), классификатор не перепутан.\n")

    # ---- 4. Контроль κ=0.1: маргиналы vs число шагов ------------------------
    A("## 4. Контроль κ=0.1 (маргиналы проваливались 5σ): E и маргиналы vs steps\n")
    A("| steps | max|E| | max|P(s+)−0.5| (в σ) |")
    A("|---|---|---|")
    for steps in (6000, 24000, 48000):
        counts = sweep(base_cell(0.1, steps=steps), COARSE, [0])
        E = analysis.E_from_counts(counts)
        p_s, p_t = analysis.marginals(counts)
        n = counts.sum(-1)
        sig = analysis.marginal_sigma(p_s, n)
        max_marg_sigma = float(np.max(np.abs(p_s - 0.5) / sig))
        A(f"| {steps} | {np.max(np.abs(E)):.4f} | {max_marg_sigma:.2f} |")
    A("\nЕсли отклонение маргиналов падает с ростом steps — это НЕ-сходимость "
      "рыхлой слабосвязанной цепи, а не баг классификатора (κ=1,10 маргиналы проходят).\n")

    A(f"\n---\nВремя протокола: {time.time()-t_all:.1f} с.")
    out = ROOT / "results" / "R1" / "paranoia.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"paranoia → {out}")
    print(f"  шаги-дрейф κ10: {drift:.4f}; удвоение N,B: {dmax:.4f}; p̂_big={p_big:.3f} ({'ферро' if sgn_big<0 else 'антиферро'})")


if __name__ == "__main__":
    main()
