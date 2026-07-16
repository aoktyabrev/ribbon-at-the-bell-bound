"""C2-TM этап 2 анализ ПОСЛЕ коммита сырья. KILL по prereg 97b9622:
«ни одна форма ≤2 параметров не даёт χ²/dof<2 на объединённых ≥6 точках С
попаданием held-out предсказания в 3σ ⇒ простой замкнутой формы нет».

Held-out предсказания зафиксированы в C2TM_predictions.md (99ad201) из фита
на 4 замороженных точках. Здесь: (1) сверка измеренных A(0.25),A(8) с этими
предсказаниями (3σ_pred); (2) переобучение M1-M3 на объединённых точках, χ²/dof.
Код написан ДО сырья этапа 2. Формы закрыты (M4 вычеркнута).
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

# замороженные 4 (a9cef7b)
KF0 = [0.5, 1.0, 2.0, 4.0]
A0 = [0.27, 0.41833333333333333, 0.555, 0.8366666666666667]
S0 = [0.02779538330970331, 0.02622017706243009, 0.024013451091697197, 0.01581109549464365]

M1 = lambda k, c, g: np.tanh(c * k**g)
M2 = lambda k, g, k0: k**g / (k**g + k0)
M3 = lambda k, c, g: 1.0 - np.exp(-c * k**g)
FORMS = {"M1: tanh(c·k^γ)": (M1, [0.4, 0.6]),
         "M2: k^γ/(k^γ+k0)": (M2, [0.8, 1.0]),
         "M3: 1−exp(−c·k^γ)": (M3, [0.4, 0.6])}
HELDOUT = [0.25, 8.0]


def main():
    d = json.load(open(os.path.join(RES, "C2TM_stage2_raw.json")))
    pred = json.load(open(os.path.join(RES, "C2TM_stage1_fits.json")))["rows"]
    pred = {r["name"]: r["preds"] for r in pred}

    pts = d["points"]
    new = {float(v["k_f"]): (v["A"], v["sigma"]) for v in pts.values()}
    print("=== C2-TM этап 2: измеренные новые точки ===")
    for k in sorted(new):
        print(f"  A({k}) = {new[k][0]:.4f} ± {new[k][1]:.4f}")

    # объединённая выборка (frozen 4 + новые)
    KF = list(KF0) + [k for k in sorted(new)]
    A = list(A0) + [new[k][0] for k in sorted(new)]
    S = list(S0) + [new[k][1] for k in sorted(new)]
    KF, A, S = np.array(KF), np.array(A), np.array(S)
    n = len(KF)

    print("\n=== KILL-тест: χ²/dof<2 на объединённых И held-out в 3σ ===")
    out = {"meta": dict(raw="C2TM_stage2_raw.json", prereg=d["meta"]["prereg_commit"],
                        n_points=n), "forms": {}}
    survivors = []
    for name, (f, p0) in FORMS.items():
        popt, _ = curve_fit(f, KF, A, p0=p0, sigma=S, absolute_sigma=True, maxfev=20000)
        chi2 = float(np.sum(((A - f(KF, *popt)) / S)**2))
        dof = n - len(popt)
        chi2_dof = chi2 / dof
        # held-out: измерено vs ПРЕДСКАЗАНО (из 4-точечного фита, 99ad201)
        ho = {}
        ho_ok = True
        for k in HELDOUT:
            if k in new:
                pv, ps = pred[name][str(k)] if str(k) in pred[name] else pred[name][f"{k}"]
                meas, ms = new[k]
                sig = np.sqrt(ps**2 + ms**2)
                dev = abs(meas - pv) / max(sig, 1e-9)
                ho[k] = dict(pred=pv, pred_sig=ps, meas=meas, meas_sig=ms, dev_sigma=float(dev),
                             within3=bool(dev <= 3))
                ho_ok = ho_ok and dev <= 3
        survives = (chi2_dof < 2) and ho_ok
        survivors.append(survives) if survives else None
        out["forms"][name] = dict(popt=[float(x) for x in popt], chi2=chi2, dof=dof,
                                  chi2_dof=chi2_dof, heldout=ho, survives=bool(survives))
        hostr = "  ".join(f"A({k}):{v['dev_sigma']:.1f}σ{'✓' if v['within3'] else '✗'}" for k, v in ho.items())
        print(f"  {name}: χ²/dof={chi2_dof:.2f} ({'✓<2' if chi2_dof<2 else '✗≥2'})  "
              f"held-out[{hostr}]  ⇒ {'ВЫЖИЛА' if survives else 'отпала'}")

    kill = not any(s["survives"] for s in out["forms"].values())
    out["kill_fired"] = bool(kill)
    verdict = ("KILL: простой замкнутой формы (≤2 пар.) НЕТ — A(k_f) остаётся "
               "кинетическим числом" if kill else
               "НЕ kill: есть выжившая форма (χ²/dof<2 И held-out в 3σ)")
    print(f"\n  ⇒ {verdict}")
    json.dump(out, open(os.path.join(RES, "C2TM_stage2_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2TM_stage2_analysis.json")


if __name__ == "__main__":
    main()
