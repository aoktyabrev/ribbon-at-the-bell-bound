"""D2-ext. Анализ ПОСЛЕ коммита сырья (D2ext_prereg §3).

A_N=|E(0)|, F_N=|E(π/4)|, ρ_N=F_N/A_N (флип t̃=−t не меняет |E|). Фит A_N vs N:
M0 (const A∞), M1 (A·N^−γ), M2 (A∞+c·N^−γ). Сравнение по AICc (n=5 мало ⇒ поправка).
Вердикт D2ext-H1. ρ_N — монотонность (таблица+σ). Фигуры + D2ext_analysis.json.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "fig")


def load():
    with open(os.path.join(RES, "D2ext_raw_tables.json")) as f:
        return json.load(f)


def aicc(chi2, k, n):
    aic = chi2 + 2 * k
    denom = n - k - 1
    return aic + (2 * k * (k + 1) / denom if denom > 0 else np.inf)


def main():
    d = load()
    NS = np.array(d["meta"]["NS"], float)
    A = np.array([abs(d["odd"][f"N{int(N)}|odd|0"]["E_raw"]) for N in NS])
    sA = np.array([d["odd"][f"N{int(N)}|odd|0"]["sigma"] for N in NS])
    F = np.array([abs(d["odd"][f"N{int(N)}|odd|pi/4"]["E_raw"]) for N in NS])
    sF = np.array([d["odd"][f"N{int(N)}|odd|pi/4"]["sigma"] for N in NS])
    n = len(NS)
    print("A_N=|E(0)|:", [f"{a:.3f}±{s:.3f}" for a, s in zip(A, sA)])
    print("F_N=|E(π/4)|:", [f"{f:.3f}±{s:.3f}" for f, s in zip(F, sF)])

    # --- M0: константа (взвешенное среднее) ---
    w = 1.0 / sA**2
    A0 = float(np.sum(w * A) / np.sum(w))
    A0_err = float(np.sqrt(1.0 / np.sum(w)))
    chi2_0 = float(np.sum(((A - A0) / sA)**2))
    aicc0 = aicc(chi2_0, 1, n)

    # --- M1: A·N^−γ ---
    def m1(N, Aa, g): return Aa * N**(-g)
    p1, c1 = curve_fit(m1, NS, A, p0=[1.0, 0.2], sigma=sA, absolute_sigma=True, maxfev=10000)
    chi2_1 = float(np.sum(((A - m1(NS, *p1)) / sA)**2))
    aicc1 = aicc(chi2_1, 2, n)

    # --- M2: A∞ + c·N^−γ ---
    def m2(N, Ainf, c, g): return Ainf + c * N**(-g)
    aicc2 = np.inf; p2 = None; chi2_2 = np.nan; Ainf_err = np.nan
    try:
        p2, c2 = curve_fit(m2, NS, A, p0=[0.3, 1.0, 0.5], sigma=sA,
                           absolute_sigma=True, maxfev=20000)
        chi2_2 = float(np.sum(((A - m2(NS, *p2)) / sA)**2))
        aicc2 = aicc(chi2_2, 3, n)
        Ainf_err = float(np.sqrt(abs(c2[0, 0])))
    except Exception as e:
        print("  M2 фит не сошёлся:", e)

    models = {"M0_const": dict(aicc=aicc0, chi2=chi2_0, k=1, A_inf=A0, A_inf_err=A0_err),
              "M1_power": dict(aicc=aicc1, chi2=chi2_1, k=2, A=float(p1[0]), gamma=float(p1[1])),
              "M2_satur": dict(aicc=aicc2, chi2=chi2_2, k=3,
                               A_inf=(float(p2[0]) if p2 is not None else None),
                               c=(float(p2[1]) if p2 is not None else None),
                               gamma=(float(p2[2]) if p2 is not None else None),
                               A_inf_err=(Ainf_err if p2 is not None else None))}
    best = min(models, key=lambda m: models[m]["aicc"])
    aiccs = {m: models[m]["aicc"] for m in models}
    print("=== фиты A_N vs N ===")
    for m, v in models.items():
        if v.get("A_inf") is not None:
            extra = f"A∞={v['A_inf']:.3f}±{v['A_inf_err']:.3f}" + (
                f" c={v['c']:.3f} γ={v['gamma']:.3f}" if v.get("c") is not None else "")
        elif "A" in v:
            extra = f"A={v['A']:.3f} γ={v['gamma']:.3f}"
        else:
            extra = ""
        print(f"  {m}: AICc={v['aicc']:.2f} χ²={v['chi2']:.2f}  {extra}")
    # ΔAICc относительно лучшей
    dA = {m: aiccs[m] - aiccs[best] for m in models}
    print("  ΔAICc:", {m: round(v, 2) for m, v in dA.items()})

    # --- вердикт D2ext-H1 ---
    plateau_models = ["M0_const", "M2_satur"]
    m1_delta = dA["M1_power"]
    # A∞ ≥ 3σ от нуля (по выигравшей плато-модели)
    if best in plateau_models:
        Ainf = models[best]["A_inf"]; Ainf_e = models[best]["A_inf_err"]
        sig3 = (Ainf is not None and Ainf_e is not None and Ainf >= 3 * Ainf_e)
    else:
        sig3 = False
    win_margin = min(dA[m] for m in models if m != best)  # насколько лучшая опережает 2-ю
    if best in plateau_models and win_margin >= 4 and sig3:
        verdict = f"D2ext-H1 ПОДТВЕРЖДЁН: плато {best} (ΔAICc≥4, A∞≥3σ) — неразбавляемо"
    elif best == "M1_power" and win_margin >= 4:
        verdict = "KILL: M1 (степенной спад к 0) выигрывает ΔAICc≥4 — рецидив R4e"
    else:
        verdict = f"НЕ РАЗРЕШЕНО на доступных N (лучшая {best}, но ΔAICc<4 или A∞<3σ)"
    print(f"  ⇒ {verdict}")

    # --- ρ_N = F_N/A_N (форма) ---
    rho = F / A
    srho = rho * np.sqrt((sF / F)**2 + (sA / A)**2)
    print("=== ρ_N = F_N/A_N (форма) ===")
    for N, r, s in zip(NS, rho, srho):
        print(f"  N{int(N)}: ρ={r:.3f}±{s:.3f}")
    monotone_up = all(rho[i+1] - rho[i] > 0 for i in range(n-1))
    monotone_up_sig = all(rho[i+1] - rho[i] > (srho[i]+srho[i+1]) for i in range(n-1))
    rho_verdict = ("монотонный рост ЗА пределами σ (форма эволюционирует)" if monotone_up_sig
                   else "монотонный рост в пределах σ (слабый намёк)" if monotone_up
                   else "без монотонного тренда (форма стабильна; рост CHSH в D2 — шум)")
    print(f"  ⇒ ρ_N: {rho_verdict}")

    out = dict(A_N=list(map(float, A)), sigma_A=list(map(float, sA)),
               F_N=list(map(float, F)), sigma_F=list(map(float, sF)),
               NS=list(map(int, NS)), models=models, best=best,
               delta_aicc={m: float(v) for m, v in dA.items()},
               verdict_H1=verdict, rho_N=list(map(float, rho)),
               sigma_rho=list(map(float, srho)), rho_verdict=rho_verdict)
    with open(os.path.join(RES, "D2ext_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    # --- фигуры ---
    plt.figure(figsize=(7, 5))
    plt.errorbar(NS, A, yerr=sA, fmt="o", capsize=3, label="A_N=|E(0)| (данные)", color="C0", zorder=5)
    Nf = np.linspace(NS.min(), NS.max(), 100)
    plt.axhline(A0, ls=":", color="C1", label=f"M0 const A∞={A0:.3f}")
    plt.plot(Nf, m1(Nf, *p1), "--", color="C2", label=f"M1 A·N^−γ (γ={p1[1]:.2f})")
    if p2 is not None:
        plt.plot(Nf, m2(Nf, *p2), "-", color="C3", label=f"M2 A∞+cN^−γ (A∞={p2[0]:.3f})")
    plt.xlabel("N"); plt.ylabel("A_N"); plt.ylim(0, max(A)*1.3)
    plt.title(f"D2-ext скейлинг: лучшая {best} (M=1200)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "d2ext_scaling.png"), dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.errorbar(NS, rho, yerr=srho, fmt="s-", capsize=3, color="C4")
    plt.xlabel("N"); plt.ylabel("ρ_N = F_N/A_N")
    plt.title("D2-ext трек формы ρ_N")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "d2ext_rho.png"), dpi=130); plt.close()
    print(f"  анализ → {RES}/D2ext_analysis.json ; фигуры → {FIG}/d2ext_*.png")


if __name__ == "__main__":
    main()
