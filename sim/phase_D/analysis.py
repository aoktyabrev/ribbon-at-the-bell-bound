"""D2-slim. Анализ ПОСЛЕ фиксации и коммита сырых таблиц (D2_prereg §4).

ПОРЯДОК (нарушение = невалидность стадии): сырьё (run_grid) → КОММИТ → этот анализ.
Здесь: конвенционный флип t̃=−t (A–C: E→−E); фиты формы нечётного сектора
(−A·cosθ vs −tanh(β·cosθ)); скейлинг A_N vs N; CHSH. Фигуры + D2_analysis.json.

Запуск (ТОЛЬКО после коммита сырья):
    JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/analysis.py
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

THETA_MAP = {"0": 0.0, "pi/8": np.pi/8, "pi/4": np.pi/4, "3pi/8": 3*np.pi/8,
             "pi/2": np.pi/2, "5pi/8": 5*np.pi/8, "3pi/4": 3*np.pi/4,
             "7pi/8": 7*np.pi/8, "pi": np.pi}


def load():
    with open(os.path.join(RES, "D2_raw_tables.json")) as f:
        return json.load(f)


def series(data, N, sector, flip=True):
    """E(θ) для (N,sector). flip=True — конвенционный t̃=−t ⇒ E→−E (A–C)."""
    ax = data["AX"]
    ths, Es, sigs = [], [], []
    for thn, thv in THETA_MAP.items():
        cell = f"N{N}|{sector}|{thn}"
        if cell in ax:
            ths.append(thv)
            Es.append((-1.0 if flip else 1.0) * ax[cell]["E_raw"])
            sigs.append(ax[cell]["sigma"])
    order = np.argsort(ths)
    return np.array(ths)[order], np.array(Es)[order], np.array(sigs)[order]


def fit_forms(th, E, sig):
    """Фиты E(θ) vs −A·cosθ и vs −tanh(β·cosθ). Возврат параметры + χ²/AIC."""
    c = np.cos(th)
    # −A cosθ
    def f_cos(x, A): return -A * np.cos(x)
    pc, _ = curve_fit(f_cos, th, E, p0=[0.5], sigma=sig, absolute_sigma=True)
    res_c = (E - f_cos(th, *pc)) / sig
    chi2_c = float(np.sum(res_c**2))
    # −tanh(β cosθ)
    def f_tanh(x, beta): return -np.tanh(beta * np.cos(x))
    pt, _ = curve_fit(f_tanh, th, E, p0=[0.5], sigma=sig, absolute_sigma=True)
    res_t = (E - f_tanh(th, *pt)) / sig
    chi2_t = float(np.sum(res_t**2))
    k = 1  # оба однопараметрические
    n = len(th)
    aic_c = chi2_c + 2*k
    aic_t = chi2_t + 2*k
    return dict(A_cos=float(pc[0]), chi2_cos=chi2_c,
                beta_tanh=float(pt[0]), amp_tanh=float(np.tanh(pt[0])), chi2_tanh=chi2_t,
                aic_cos=aic_c, aic_tanh=aic_t,
                closer_to="tanh" if aic_t < aic_c else "cos", n=n)


def chsh(data, N, sector="odd", flip=True):
    """CHSH из совместных таблиц: a∈{0,π/2}, b∈{π/4,3π/4}; изотропия E=E(|a−b|).
    S = E(π/4) − E(3π/4) + E(π/4) + E(π/4) = 3E(π/4) − E(3π/4). Без оптимизации углов."""
    ax = data["AX"]
    def Eth(thn):
        c = ax[f"N{N}|{sector}|{thn}"]
        return (-1.0 if flip else 1.0) * c["E_raw"], c["sigma"]
    e14, s14 = Eth("pi/4")
    e34, s34 = Eth("3pi/4")
    S = 3*e14 - e34
    sigS = float(np.sqrt((3*s14)**2 + s34**2))
    return dict(S=float(S), absS=float(abs(S)), sigma=sigS,
                E_pi4=e14, E_3pi4=e34, angles="a∈{0,π/2}, b∈{π/4,3π/4}")


def main():
    data = load()
    NS = data["meta"]["NS"]
    out = {"convention": "t̃=−t применён (E→−E), A–C", "forms": {}, "scaling": {}, "chsh": {}}

    # --- фиты формы нечётного сектора по N ---
    print("=== D2-H2 форма (нечёт, флип t̃=−t) ===")
    for N in NS:
        th, E, sig = series(data, N, "odd")
        fr = fit_forms(th, E, sig)
        out["forms"][f"N{N}"] = fr
        print(f"  N{N}: A_cos={fr['A_cos']:.3f} χ²={fr['chi2_cos']:.2f} | "
              f"β_tanh={fr['beta_tanh']:.3f} amp={fr['amp_tanh']:.3f} χ²={fr['chi2_tanh']:.2f} | "
              f"ближе к: {fr['closer_to']}")

    # --- скейлинг A_N (амплитуда при θ=0 нечёт) ---
    print("=== D2-H1 скейлинг A_N vs N (нечёт) ===")
    A0, Api = [], []
    for N in NS:
        c0 = data["AX"][f"N{N}|odd|0"]; cpi = data["AX"][f"N{N}|odd|pi"]
        A0.append((abs(c0["E_raw"]), c0["sigma"]))
        Api.append((abs(cpi["E_raw"]), cpi["sigma"]))
        print(f"  N{N}: |E(0)|={abs(c0['E_raw']):.3f}±{c0['sigma']:.3f}  "
              f"|E(π)|={abs(cpi['E_raw']):.3f}±{cpi['sigma']:.3f}")
    # вердикт: монотонный спад за пределами σ?
    vals = [a for a, _ in A0]; sigs = [s for _, s in A0]
    monotone_down = all(vals[i] - vals[i+1] > (sigs[i]+sigs[i+1]) for i in range(len(vals)-1))
    # совместимость (все пары в пределах 2σ)
    compatible = all(abs(vals[i]-vals[j]) < 2*(sigs[i]+sigs[j])
                     for i in range(len(vals)) for j in range(i+1, len(vals)))
    verdict = ("KILL: монотонный спад A_N (конечно-размерный, рецидив R4e)" if monotone_down
               else "D2-H1 ПОДТВЕРЖДЁН: A_N совместимы между N (нет тренда вниз)"
               if compatible else "неоднозначно (не монотонно, но и не строго совместимо)")
    out["scaling"] = dict(A0=A0, Api=Api, monotone_down=bool(monotone_down),
                          compatible=bool(compatible), verdict=verdict)
    print(f"  ⇒ {verdict}")

    # --- CHSH ---
    print("=== D2-H3 CHSH (нечёт) ===")
    for N in NS:
        ch = chsh(data, N, "odd")
        out["chsh"][f"N{N}"] = ch
        flag = "S≤2 ✓" if ch["absS"] <= 2.0 + ch["sigma"] else "S>2 — АУДИТ!"
        print(f"  N{N}: |S|={ch['absS']:.3f}±{ch['sigma']:.3f}  ({flag})")

    with open(os.path.join(RES, "D2_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    # --- фигуры ---
    plt.figure(figsize=(7.5, 5))
    colors = {16: "C0", 32: "C1", 48: "C2"}
    for N in NS:
        th, E, sig = series(data, N, "odd")
        plt.errorbar(th, E, yerr=sig, fmt="o", color=colors.get(N, "k"),
                     label=f"N={N} (odd, raw)", capsize=2)
        fr = out["forms"][f"N{N}"]
        thf = np.linspace(0, np.pi, 100)
        plt.plot(thf, -fr["A_cos"]*np.cos(thf), "--", color=colors.get(N, "k"), alpha=0.5)
        plt.plot(thf, -np.tanh(fr["beta_tanh"]*np.cos(thf)), "-", color=colors.get(N, "k"), alpha=0.8)
    plt.xlabel("θ (angle between axes)"); plt.ylabel("E(θ) [flip t̃=−t]")
    plt.title("D2: E(θ) odd sector + fits (--=−A·cosθ, —=−tanh(β·cosθ))")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "d2_Etheta_odd.png"), dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    xs = NS
    y0 = [a for a, _ in out["scaling"]["A0"]]; e0 = [s for _, s in out["scaling"]["A0"]]
    plt.errorbar(xs, y0, yerr=e0, fmt="o-", capsize=3, label="A_N=|E(0)| odd")
    plt.xlabel("N"); plt.ylabel("A_N"); plt.ylim(0, 1)
    plt.title("D2-H1 scaling: A_N vs N")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "d2_scaling_AN.png"), dpi=130); plt.close()

    print(f"  анализ → {RES}/D2_analysis.json ; фигуры → {FIG}/d2_*.png")


if __name__ == "__main__":
    main()
