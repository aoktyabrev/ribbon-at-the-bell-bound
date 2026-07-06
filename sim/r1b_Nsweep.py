#!/usr/bin/env python
"""R1b (диагностика по указанию архитектора): свип N при κ=1, geodesic, T=0,
до сходимости (блочный критерий). Гипотеза: амплитуда E(0) монотонно падает
0.62 → 0.38 с ростом N (динамический интерьер); вопрос насыщения при больших N.

Результат — results/R1b/report.md. Пре-регистрацию не трогаем.
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

from ribbon_sim import analysis, plots
from ribbon_sim.experiment import run_cell_blocks

COARSE = np.radians(np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]))
N_GRID = [2, 4, 8, 16, 32, 64]
BLOCK = {"block_steps": 10000, "ceiling": 120000, "sigma_mult": 2.0, "e_floor": 0.01}


def main():
    out = ROOT / "results" / "R1b"
    out.mkdir(parents=True, exist_ok=True)
    L = []; A = L.append
    A("# R1b — свип N при κ=1 (geodesic, до сходимости)\n")
    A(f"GPU: {jax.devices()[0].device_kind}. Гипотеза: амплитуда E(0) падает 0.62→0.38 "
      "с ростом N; вопрос насыщения. Блочный критерий сходимости.\n")
    A("| N | k_e | E(0) | амп max\\|E\\| | A1 | A3 | A3/A1 | сошлось% | max_steps |")
    A("|---|---|---|---|---|---|---|---|---|")
    t0 = time.time()
    rows = []
    for N in N_GRID:
        k_e = 1.0 * (N - 1) * 1.0
        cell = {"N": N, "k_e": k_e, "k_c": 1.0, "spinor": False, "elastic": "geodesic",
                "T0": 0.0, "decay": 1.0, "steps": BLOCK["block_steps"], "B": 8192,
                "seeds": [0, 1], "lr": 0.5 / (k_e + 1.0), "label": f"N={N}"}
        res = run_cell_blocks(cell, COARSE, BLOCK)
        E = analysis.E_from_counts(res["counts"].sum(0))
        h = analysis.harmonics(COARSE, E)
        a3a1 = h["A3"] / h["A1"] if abs(h["A1"]) > 1e-6 else float("nan")
        E0 = float(E[0]); amp = float(np.max(np.abs(E)))
        rows.append((N, E0, amp))
        A(f"| {N} | {k_e:.0f} | {E0:+.4f} | {amp:.4f} | {h['A1']:+.3f} | {h['A3']:+.3f} "
          f"| {a3a1:+.3f} | {res['frac_converged']*100:.0f} | {res['max_steps']} |")
        print(f"  N={N}: E(0)={E0:+.4f} amp={amp:.4f} conv={res['frac_converged']*100:.0f}%")

    # график амплитуды vs N
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = [r[0] for r in rows]; E0s = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ns, E0s, "o-")
    ax.set_xscale("log", base=2); ax.set_xlabel("N (log2)"); ax.set_ylabel("E(0) = ампл. при θ=0")
    ax.set_title("R1b: амплитуда корреляции vs длина ленты (κ=1)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / "amp_vs_N.png", dpi=120)
    plt.close(fig)
    A(f"\n![амплитуда vs N](amp_vs_N.png)\n")

    e0_first, e0_last = rows[0][1], rows[-1][1]
    A(f"**Тренд E(0):** N={rows[0][0]} → {e0_first:+.3f} … N={rows[-1][0]} → {e0_last:+.3f}. "
      f"{'Монотонное падение ✅' if e0_first > e0_last else 'НЕ монотонно ❌'} "
      f"(гипотеза 0.62→0.38).")
    A(f"\nВремя: {time.time()-t0:.1f} с.")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"R1b → {out / 'report.md'}")


if __name__ == "__main__":
    main()
