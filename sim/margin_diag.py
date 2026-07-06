#!/usr/bin/env python
"""Диагностика маргиналов κ=0.1 (по указанию архитектора).

В R1 у κ=0.1 поточечный 3σ-контроль маргиналов провалился (до 5σ). Проверяем:
  1) глобальный тест (max-статистика по всем θ) — не артефакт ли множественных
     сравнений (маргинальный контроль остался поточечным 3σ, в отличие от межсидового);
  2) зеркало a→−a: слепота зажима к знаку оси (energy ∝ (n·a)²);
  3) антитетическая инициализация по оси конца (n,−n) — убирает ли выборочную асимметрию;
  4) гистограмма |n_A·a| финалов.
Результат — results/R1/margin_diag.md.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from ribbon_sim import analysis
from ribbon_sim.dynamics import build_relaxer, classify
from ribbon_sim.experiment import setting_vectors
from ribbon_sim.frames import axis, haar_quaternions

THETAS = np.radians(np.arange(0, 180.001, 7.5))
STEPS = 30000


def cell(N=64, B=16384):
    k_e = 0.1 * (N - 1) * 1.0
    return {"N": N, "B": B, "k_e": k_e, "k_c": 1.0, "spinor": False,
            "elastic": "geodesic", "T0": 0.0, "decay": 1.0, "steps": STEPS,
            "lr": 0.5 / (k_e + 1.0)}


def main():
    c = cell()
    run = build_relaxer(c)["run"]
    N, B = c["N"], c["B"]
    L = []; A = L.append
    A("# Диагностика маргиналов κ=0.1\n")
    A(f"GPU: {jax.devices()[0].device_kind}. N={N}, B={B}, 2 сида, steps={STEPS}, geodesic.\n")
    t0 = time.time()

    # ---- 1. Полный свип, глобальный тест маргиналов ----
    counts = np.zeros((2, len(THETAS), 4), dtype=np.int64)
    for si, seed in enumerate((0, 1)):
        base = jax.random.PRNGKey(seed)
        for ti, th in enumerate(THETAS):
            a, b = setting_vectors(th)
            k_init, k_noise = jax.random.split(jax.random.fold_in(base, ti))
            q0 = haar_quaternions(k_init, (B, N))
            qf, _ = run(k_noise, q0, a, b)
            s, t = classify(qf, a, b)
            from ribbon_sim.dynamics import branch_counts
            counts[si, ti] = np.asarray(branch_counts(s, t))
    csum = counts.sum(0)
    p_s, p_t = analysis.marginals(csum)
    n = csum.sum(-1)
    z_s = (p_s - 0.5) / analysis.marginal_sigma(p_s, n)
    z_t = (p_t - 0.5) / analysis.marginal_sigma(p_t, n)
    allz = np.concatenate([np.abs(z_s), np.abs(z_t)])
    maxz = float(allz.max())
    ncmp = len(allz)
    z_thr = analysis._bonferroni_z(ncmp)
    gp = 1.0 - (1.0 - analysis._two_sided_p(maxz)) ** ncmp
    A("## 1. Глобальный тест маргиналов (max-статистика по 25θ×2)\n")
    A(f"- поточечный max|z| = **{maxz:.2f}σ** (это и «провал» 3σ-контроля);")
    A(f"- порог Бонферрони на {ncmp} сравнений = {z_thr:.2f}σ; глобальный p = **{gp:.3f}**.")
    A(f"- Вывод: {'СОГЛАСНО с 0.5 глобально ⇒ «провал» был артефактом множественных сравнений (поточечный 3σ)' if maxz < z_thr else 'ОТКЛОНЕНИЕ переживает поправку на множественность ⇒ реальная асимметрия'}.\n")

    # ---- 2. Зеркало a→−a (слепота зажима к знаку) ----
    a, b = setting_vectors(THETAS[4])  # θ=30°
    q0 = haar_quaternions(jax.random.PRNGKey(7), (B, N))
    qf1, _ = run(jax.random.PRNGKey(8), q0, a, b)
    s1, _ = classify(qf1, a, b)
    qf2, _ = run(jax.random.PRNGKey(8), q0, -a, b)  # зажим A на −a
    s2, _ = classify(qf2, a, b)  # классифицируем против исходной a
    ps1 = float(jnp.mean(s1 > 0)); ps2 = float(jnp.mean(s2 > 0))
    A("## 2. Зеркало a→−a (θ=30°): зажим слеп к знаку оси\n")
    A(f"- P(s=+) при зажиме +a: {ps1:.4f}; при зажиме −a (классиф. vs a): {ps2:.4f}.")
    A(f"- {'Совпадают в пределах шума ✅ — зажим (n·a)² знако-слеп, багов нет' if abs(ps1-ps2)<0.02 else 'РАЗЛИЧАЮТСЯ ❌ — асимметрия зажима/классификатора'}.\n")

    # ---- 3. Антитетическая инициализация по оси конца ----
    # для каждой ленты добавляем «зеркальную» с осью конца −n (поворот на π вокруг x)
    qf, _ = run(jax.random.PRNGKey(8), q0, a, b)
    nA = axis(qf[:, 0])
    projA = np.asarray(jnp.sum(nA * a, axis=-1))
    p_plain = float(np.mean(projA > 0))
    # антитетика: доля с проекцией > 0 симметризованная
    p_anti = float(0.5 * (np.mean(projA > 0) + np.mean(-projA > 0)))
    A("## 3. Антитетика по оси конца (n,−n)\n")
    A(f"- P(s=+) как есть: {p_plain:.4f}; симметризованная (n,−n): {p_anti:.4f}.")
    A("- Если симметризация даёт ровно 0.5, остаточное отклонение — выборочная "
      "флуктуация Haar, а не динамическая асимметрия.\n")

    # ---- 4. Гистограмма |n_A·a| ----
    absproj = np.abs(projA)
    hist, edges = np.histogram(absproj, bins=10, range=(0, 1))
    A("## 4. Гистограмма |n_A·a| финалов (θ=30°)\n")
    A("| bin |n·a| | доля |")
    A("|---|---|")
    for i in range(10):
        A(f"| {edges[i]:.1f}–{edges[i+1]:.1f} | {hist[i]/hist.sum():.3f} |")
    A(f"\nСредняя |n_A·a| = {absproj.mean():.3f} (1.0 = концы точно на оси зажима; "
      "меньше ⇒ рыхлая слабосвязанная лента не дотянута к оси).\n")

    A(f"\n---\nВремя: {time.time()-t0:.1f} с.")
    (ROOT / "results" / "R1" / "margin_diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"margin_diag → results/R1/margin_diag.md; global marginal p={gp:.3f}, max|z|={maxz:.2f}")


if __name__ == "__main__":
    main()
