"""C2-ISO-DYN анализ ПОСЛЕ коммита сырья. H-I0 (амплитуда, дуэль),
H-I1 (форма: треугольник vs косинус vs ни-то-ни-другое, AICc),
H-I2 (S≤2). Только зарегистрированные ветки. Код до сырья.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")
PRIORS_HI0 = {"1.0": dict(arch=(0.19, 0.06), free=0.164),
              "4.0": dict(arch=(0.29, 0.10), free=0.336)}


def aicc(chi2, k, n):
    d = n - k - 1
    return chi2 + 2*k + (2*k*(k+1)/d if d > 0 else np.inf)


def main():
    d = json.load(open(os.path.join(RES, "C2ISO_raw.json")))
    out = {"meta": dict(raw="C2ISO_raw.json", prereg=d["meta"]["prereg_commit"],
                        tcov1=d["meta"].get("tcov1"), tcov2=d["meta"].get("tcov2")), "cells": {}}
    print("=== C2-ISO-DYN: H-I0/H-I1/H-I2 ===")
    for cellname, cell in d["cells"].items():
        kf = str(cell["kf"])
        th = np.array([cell["theta"][k]["theta"] for k in cell["theta"]])
        E = np.array([cell["theta"][k]["E"] for k in cell["theta"]])
        sg = np.array([cell["theta"][k]["sigma"] for k in cell["theta"]])
        n = len(th)
        # H-I1: треугольник ρ(1−2θ/π) vs косинус ρcosθ
        ftri = lambda x, r: r * (1 - 2*x/np.pi)
        fcos = lambda x, r: r * np.cos(x)
        pr, _ = curve_fit(ftri, th, E, p0=[E[0]], sigma=sg, absolute_sigma=True)
        pc, _ = curve_fit(fcos, th, E, p0=[E[0]], sigma=sg, absolute_sigma=True)
        chi_tri = float(np.sum(((E - ftri(th, *pr))/sg)**2))
        chi_cos = float(np.sum(((E - fcos(th, *pc))/sg)**2))
        a_tri, a_cos = aicc(chi_tri, 1, n), aicc(chi_cos, 1, n)
        dA = a_cos - a_tri            # >0 ⇒ треугольник лучше
        if abs(dA) < 4:
            shape = "ни-то-ни-другое (ΔAICc<4)"
        else:
            shape = "триангл" if dA > 0 else "косинус"
        rho = abs(float(pr[0]))       # амплитуда триангла
        rho_cos = abs(float(pc[0]))
        # H-I0 дуэль
        pri = PRIORS_HI0[kf]
        arch_dev = abs(rho - pri["arch"][0]) / pri["arch"][1]
        # H-I2: CHSH
        S, sS = cell["chsh"]["S"], cell["chsh"]["sigma"]
        s_ok = abs(S) <= 2 + sS
        out["cells"][cellname] = dict(
            kf=cell["kf"], rho_triangle=rho, rho_cos=rho_cos,
            chi2_tri=chi_tri, chi2_cos=chi_cos, aicc_tri=a_tri, aicc_cos=a_cos,
            dAICc_cos_minus_tri=dA, H_I1_shape=shape,
            H_I0=dict(rho_measured=rho, arch=pri["arch"], arch_dev_sigma=float(arch_dev),
                      model_free=pri["free"]),
            H_I2=dict(S=S, sigma=sS, S_le_2=bool(s_ok),
                      verdict="S≤2 ✓" if s_ok else "S>2 — СТОП-АУДИТ (Пирл/симметрия/no-signaling первыми)"))
        print(f"  k_f×{cell['kf']}:")
        print(f"    H-I0 ρ_dyn={rho:.4f} | архитектор {pri['arch'][0]}±{pri['arch'][1]} ({arch_dev:.1f}σ) "
              f"| модель-free {pri['free']}")
        print(f"    H-I1 форма: {shape} (χ²_tri={chi_tri:.1f} χ²_cos={chi_cos:.1f} ΔAICc={dA:+.1f})")
        print(f"    H-I2 |S|={abs(S):.4f}±{sS:.4f}  {out['cells'][cellname]['H_I2']['verdict']}")
    json.dump(out, open(os.path.join(RES, "C2ISO_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2ISO_analysis.json")


if __name__ == "__main__":
    main()
