"""C2-TM этап 1 (БЕЗ прогонов): фит форм M1-M3 на ЗАМОРОЖЕННЫХ 4 точках S1
(a9cef7b), AICc, held-out предсказания при k_f∈{0.25, 8}. Пишет C2TM_predictions.md.
Формы зафиксированы в C2TM_prereg (97b9622); добавления запрещены.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

# Замороженные 4 точки S1runs R1 (a9cef7b)
KF = np.array([0.5, 1.0, 2.0, 4.0])
A = np.array([0.27, 0.41833333333333333, 0.555, 0.8366666666666667])
SIG = np.array([0.02779538330970331, 0.02622017706243009,
                0.024013451091697197, 0.01581109549464365])
HELDOUT = [0.25, 8.0]

M1 = lambda k, c, g: np.tanh(c * k**g)
M2 = lambda k, g, k0: k**g / (k**g + k0)
M3 = lambda k, c, g: 1.0 - np.exp(-c * k**g)
FORMS = {"M1: tanh(c·k^γ)": (M1, [0.4, 0.6]),
         "M2: k^γ/(k^γ+k0)": (M2, [0.8, 1.0]),
         "M3: 1−exp(−c·k^γ)": (M3, [0.4, 0.6])}


def aicc(chi2, k, n):
    d = n - k - 1
    return chi2 + 2*k + (2*k*(k+1)/d if d > 0 else np.inf)


def pred_with_sigma(f, popt, pcov, k):
    """Предсказание f(k) и σ через якобиан (численно)."""
    val = float(f(k, *popt))
    J = np.zeros(len(popt))
    for i in range(len(popt)):
        dp = np.zeros(len(popt)); h = 1e-6 * (abs(popt[i]) + 1e-6); dp[i] = h
        J[i] = (f(k, *(popt+dp)) - f(k, *(popt-dp))) / (2*h)
    var = float(J @ pcov @ J)
    return val, float(np.sqrt(max(var, 0.0)))


def main():
    n = len(KF)
    rows = []
    for name, (f, p0) in FORMS.items():
        popt, pcov = curve_fit(f, KF, A, p0=p0, sigma=SIG, absolute_sigma=True, maxfev=20000)
        resid = (A - f(KF, *popt)) / SIG
        chi2 = float(np.sum(resid**2))
        ac = aicc(chi2, len(popt), n)
        preds = {k: pred_with_sigma(f, popt, pcov, k) for k in HELDOUT}
        rows.append(dict(name=name, popt=[float(x) for x in popt], chi2=chi2,
                         chi2_dof=chi2/(n-len(popt)), aicc=ac, preds=preds))
        print(f"{name}: popt={[f'{x:.4f}' for x in popt]} χ²={chi2:.2f} "
              f"χ²/dof={chi2/(n-len(popt)):.2f} AICc={ac:.2f}")
        for k in HELDOUT:
            v, s = preds[k]
            print(f"    A({k}) = {v:.4f} ± {s:.4f}")

    best = min(rows, key=lambda r: r["aicc"])["name"]

    # markdown с предсказаниями (коммитится ДО прогонов этапа 2)
    lines = ["# C2-TM этап 1 — фиты M1-M3 и held-out предсказания",
             "",
             "Замороженные 4 точки S1runs R1 (a9cef7b): "
             + ", ".join(f"k_f={k}→A={a:.3f}±{s:.3f}" for k, a, s in zip(KF, A, SIG)),
             "Формы из prereg 97b9622 (M4 вычеркнута). Коммит ДО прогонов этапа 2.",
             "",
             "## Фиты (взвешенный МНК, absolute_sigma)", "",
             "| форма | параметры | χ² | χ²/dof | AICc |",
             "|---|---|---|---|---|"]
    for r in rows:
        pp = ", ".join(f"{x:.4f}" for x in r["popt"])
        star = " ★" if r["name"] == best else ""
        lines.append(f"| {r['name']}{star} | {pp} | {r['chi2']:.2f} | {r['chi2_dof']:.2f} | {r['aicc']:.2f} |")
    lines += ["", f"Лучшая по AICc: **{best}** (при n=4,k=2 dof=1 ⇒ AICc-ранг = χ²-ранг).",
              "", "## Held-out ПРЕДСКАЗАНИЯ (регистрируются здесь, до прогонов)", "",
              "| форма | A(0.25) | A(8) |", "|---|---|---|"]
    for r in rows:
        p025 = r["preds"][0.25]; p8 = r["preds"][8.0]
        lines.append(f"| {r['name']} | {p025[0]:.4f} ± {p025[1]:.4f} | {p8[0]:.4f} ± {p8[1]:.4f} |")
    lines += ["",
              "Прогноз архитектора (журнал, этап 0): A(8)=0.95±0.03, A(0.25)=0.16±0.04.",
              "",
              "KILL (prereg): ни одна форма не даёт χ²/dof<2 на объединённых ≥6 точках "
              "С попаданием held-out в 3σ ⇒ простой замкнутой формы нет.", ""]
    with open(os.path.join(HERE, "C2TM_predictions.md"), "w") as fmd:
        fmd.write("\n".join(lines))
    json.dump({"rows": rows, "best": best}, open(os.path.join(RES, "C2TM_stage1_fits.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"  → {HERE}/C2TM_predictions.md ; {RES}/C2TM_stage1_fits.json")


if __name__ == "__main__":
    main()
