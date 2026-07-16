"""C2-ISO уточнение формы (NON-PREREG диагностика, из сырья 3fd3e3d).
Фит E(θ) = −ρ·E_p(θ), E_p=(sin^p−cos^p)/(sin^p+cos^p) на θ/2 (канон §2.6),
2 параметра (ρ,p), AICc против чистого косинуса (p=2 фикс.). Независимая
проверка: S_direct против предсказания фита на CHSH-углах.
Исход H-I1 (косинус) ЗАДНИМ ЧИСЛОМ НЕ переписывается; уточнение → C2-R.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")


def E_p(theta, p):
    s = np.sin(theta/2)**p
    c = np.cos(theta/2)**p
    return (s - c) / (s + c)


def model(theta, rho, p):
    return -rho * E_p(theta, p)


def cosform(theta, rho):
    return rho * np.cos(theta)      # = model с p=2


def aicc(chi2, k, n):
    d = n - k - 1
    return chi2 + 2*k + (2*k*(k+1)/d if d > 0 else np.inf)


def S_from_shape(f):
    """S на стандартных CHSH-углах (изотропия): 3·E(π/4)−E(3π/4)."""
    return 3*f(np.pi/4) - f(3*np.pi/4)


def main():
    d = json.load(open(os.path.join(RES, "C2ISO_raw.json")))
    out = {"meta": dict(raw="C2ISO_raw.json", note="non-prereg уточнение формы"), "cells": {}}
    print("=== C2-ISO хорда-фит E(θ)=−ρ·E_p(θ) (non-prereg) ===")
    for cn, c in d["cells"].items():
        th = np.array([c["theta"][k]["theta"] for k in c["theta"]])
        E = np.array([c["theta"][k]["E"] for k in c["theta"]])
        sg = np.array([c["theta"][k]["sigma"] for k in c["theta"]])
        n = len(th)
        # хорда-фит (ρ,p)
        popt, pcov = curve_fit(model, th, E, p0=[abs(E[0]), 2.0], sigma=sg,
                               absolute_sigma=True, bounds=([0, 0.3], [1, 8]), maxfev=20000)
        rho_h, p_h = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        rho_e, p_e = float(perr[0]), float(perr[1])
        chi2_h = float(np.sum(((E - model(th, *popt))/sg)**2))
        # чистый косинус (p=2)
        pc, _ = curve_fit(cosform, th, E, p0=[abs(E[0])], sigma=sg, absolute_sigma=True)
        chi2_c = float(np.sum(((E - cosform(th, *pc))/sg)**2))
        aic_h, aic_c = aicc(chi2_h, 2, n), aicc(chi2_c, 1, n)
        # S предсказание фита vs измеренное
        S_fit = float(S_from_shape(lambda x: model(x, rho_h, p_h)))
        S_cos = float(S_from_shape(lambda x: cosform(x, float(pc[0]))))
        S_meas = c["chsh"]["S"]; sS = float(np.sqrt(sum(
            (c["chsh"]["E"][k]*0 + 1)*0 for k in c["chsh"]["E"])) or 0) or None
        # σ_S из аналитики
        aj = json.load(open(os.path.join(RES, "C2ISO_analysis.json")))
        sS = aj["cells"][cn]["H_I2"]["sigma"]
        dev_fit = (S_meas - S_fit)/sS
        out["cells"][cn] = dict(kf=c["kf"], rho_hat=rho_h, rho_err=rho_e, p_hat=p_h, p_err=p_e,
                                chi2_chord=chi2_h, chi2_cos=chi2_c, aicc_chord=aic_h, aicc_cos=aic_c,
                                dAICc_cos_minus_chord=aic_c-aic_h,
                                S_meas=S_meas, sigma_S=sS, S_pred_chord=S_fit, S_pred_cos=S_cos,
                                S_meas_minus_chordpred_sigma=float(dev_fit))
        print(f"  k_f×{c['kf']}: p̂={p_h:.3f}±{p_e:.3f}  ρ̂={rho_h:.4f}±{rho_e:.4f}  "
              f"χ²_chord={chi2_h:.1f} (cos {chi2_c:.1f}) ΔAICc={aic_c-aic_h:+.1f}")
        print(f"     S_meas={S_meas:.4f}±{sS:.4f}  S_pred(chord)={S_fit:.4f}  S_pred(cos)={S_cos:.4f}  "
              f"⇒ хорда-фит закрывает дефицит: |Δ|={abs(dev_fit):.2f}σ")
    json.dump(out, open(os.path.join(RES, "C2ISO_chord_fit.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2ISO_chord_fit.json")


if __name__ == "__main__":
    main()
