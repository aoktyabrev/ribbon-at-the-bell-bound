"""Отрисовка Fig. 2 статьи: A_plateau(k_f) — жёсткостная память (S1-runs R1).

Только рендер из ЗАМОРОЖЕННОГО сырья results/S1runs_raw.json (коммит 704e29c,
анализ a9cef7b). Никакой динамики не запускается: JAX не импортируется.
Стиль — как у остальных фигур фазы D (analysis_ds3.py).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "S1runs_raw.json")
FIG = os.path.join(HERE, "fig")


def main():
    with open(RAW) as f:
        d = json.load(f)

    meta = d["meta"]
    # порядок точек фиксируем явно, не полагаясь на порядок ключей JSON
    mults = [0.5, 1.0, 2.0, 4.0]
    rows = [d["R1_kf"][f"kf{m}"] for m in mults]
    A = np.array([r["A"] for r in rows])
    sigma = np.array([r["sigma"] for r in rows])

    plt.figure(figsize=(7, 5))
    plt.errorbar(mults, A, yerr=sigma, fmt="o-", capsize=3, color="C0",
                 label="A_plateau = |E(0)| (S1-runs R1)")
    # значение канона при k_f×1 (D2-ext плато) — для привязки к §5.1
    plt.axhline(0.363, ls=":", alpha=0.5, color="gray")
    plt.text(1.35, 0.30, "A_plateau=0.363 — D2-ext plateau (N=16..96)", fontsize=7, color="gray")

    plt.xscale("log", base=2)
    plt.xticks(mults, [f"×{m:g}" for m in mults])
    plt.xlabel("k_f (twist stiffness, × base k_f=%.1f)" % meta["base"]["k_f"])
    plt.ylabel("A_plateau = |E(0)|")
    plt.title("S1-runs: A_plateau(k_f) — stiffness memory (N=%d, M=%d, T=%.3f)"
              % (meta["N"], meta["M"], meta["T_mid"]))
    plt.ylim(0, 1)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "s1runs_kf.png")
    plt.savefig(out, dpi=130)
    plt.close()

    for m, a, s in zip(mults, A, sigma):
        print(f"k_f×{m:<4g} A={a:.3f} ± {s:.3f}")
    print("saved:", out)


if __name__ == "__main__":
    main()
