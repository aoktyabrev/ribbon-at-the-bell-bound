"""C2-R задачи 2-3 (из замороженного сырья C2R_ladder_raw, БЕЗ прогонов).
Задача 2: починка семьи отклика — Ê_β нормирована Ê_β(0)=1 (амплитуда ρ
свободна, β — ТОЛЬКО форма), правильный знак; проверка сходимости квадратуры;
перефит трёх моделей, AICc-таблица.
Задача 3: degen-гигиена ×512 — degen-доли по θ (порог 0.2, что есть в сырье);
пер-репличные проекции НЕ сохранены ⇒ вариация порога {0.1,0.3}/исключение
невозможна из frozen raw (инструментовочный пробел, требование в C2-J).
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")


def sphere_nodes(nct, nphi):
    ct = np.linspace(-1, 1, nct); phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    CT, PH = np.meshgrid(ct, phi)
    ST = np.sqrt(np.clip(1 - CT**2, 0, 1))
    return np.stack([ST*np.cos(PH), ST*np.sin(PH), CT], -1).reshape(-1, 3)


def C_theta(theta, beta, nodes):
    """<g_β(λ·a) g_β(λ·b)>_λ, a=ê, b=axis(θ). g_β(c)=tanh(βc)/tanh(β)."""
    tb = np.tanh(beta) if beta > 1e-8 else beta
    a = np.array([0, 0, 1.0]); b = np.array([np.sin(theta), 0, np.cos(theta)])
    ca = nodes @ a; cb = nodes @ b
    ga = np.tanh(beta*ca)/tb; gb = np.tanh(beta*cb)/tb
    return float(np.mean(ga*gb))


def Ehat_beta(theta_arr, beta, nodes):
    """Нормированная форма Ê_β(θ)=C(θ)/C(0), Ê_β(0)=1."""
    c0 = C_theta(0.0, beta, nodes)
    return np.array([C_theta(t, beta, nodes)/c0 for t in np.atleast_1d(theta_arr)])


def check_quadrature():
    """Сходимость: дрейф Ê_β(π/2) при сгущении узлов <1e-4."""
    for beta in (0.5, 2.0, 5.0):
        n1 = sphere_nodes(48, 48); n2 = sphere_nodes(96, 96)
        e1 = Ehat_beta(np.pi/2, beta, n1)[0]; e2 = Ehat_beta(np.pi/2, beta, n2)[0]
        print(f"    quad β={beta}: Ê(π/2) 48²={e1:.6f} 96²={e2:.6f} дрейф={abs(e1-e2):.2e}")


def aicc(chi2, k, n):
    d = n - k - 1
    return chi2 + 2*k + (2*k*(k+1)/d if d > 0 else np.inf)


def main():
    d = json.load(open(os.path.join(RES, "C2R_ladder_raw.json")))
    nodes = sphere_nodes(64, 64)
    print("=== проверка сходимости квадратуры (задача 2) ===")
    check_quadrature()
    print("\n=== перефит лестницы: косинус / семья-отклика(норм) / хорда ===")
    print(f"{'k_f':>6} {'winner':>8} {'β̂':>6} {'p̂':>6} {'ρ_rf':>6} {'ρ_cos':>6} "
          f"{'AICc:cos/rf/ch':>22} {'degen_max':>9}")
    out = {"meta": dict(raw="C2R_ladder_raw.json", note="задачи2-3 рефит, non-run"), "cells": {}}
    for ck, cell in d["cells"].items():
        th = np.array([cell["theta"][k]["theta"] for k in cell["theta"]])
        E = np.array([cell["theta"][k]["E"] for k in cell["theta"]])
        sg = np.array([cell["theta"][k]["sigma"] for k in cell["theta"]])
        degmax = max(cell["theta"][k]["degen"] for k in cell["theta"])
        n = len(th)
        # косинус
        fcos = lambda x, r: r*np.cos(x)
        pc, _ = curve_fit(fcos, th, E, p0=[E[0]], sigma=sg, absolute_sigma=True)
        chi_c = float(np.sum(((E-fcos(th,*pc))/sg)**2))
        # семья отклика НОРМИРОВАННАЯ: E=ρ·Ê_β (ρ свободна, β форма)
        def frf(x, r, b):
            return r*Ehat_beta(x, b, nodes)
        try:
            prf, _ = curve_fit(frf, th, E, p0=[E[0], 1.0], sigma=sg, absolute_sigma=True,
                               bounds=([-2, 0.02], [2, 30]), maxfev=6000)
            chi_rf = float(np.sum(((E-frf(th,*prf))/sg)**2)); beta_h = float(prf[1]); rho_rf = float(prf[0])
        except Exception as e:
            chi_rf, beta_h, rho_rf = np.inf, np.nan, np.nan
        # хорда
        def Ep(x, p):
            s = np.sin(x/2)**p; c = np.cos(x/2)**p; return (s-c)/(s+c)
        fch = lambda x, r, p: -r*Ep(x, p)
        pch, _ = curve_fit(fch, th, E, p0=[abs(E[0]), 2.0], sigma=sg, absolute_sigma=True,
                           bounds=([0, 0.3], [2, 8]), maxfev=20000)
        chi_ch = float(np.sum(((E-fch(th,*pch))/sg)**2)); p_h = float(pch[1])
        A = {"cos": aicc(chi_c,1,n), "resp": aicc(chi_rf,2,n), "chord": aicc(chi_ch,2,n)}
        winner = min(A, key=A.get)
        out["cells"][ck] = dict(kf=cell["kf"], winner=winner, beta_hat=beta_h, p_hat=p_h,
                                rho_rf=rho_rf, rho_cos=abs(float(pc[0])), aicc=A, degen_max=degmax,
                                x512_flag="провизорно/ненадёжно (degen 0.19, порог не варьируем из frozen)"
                                          if cell["kf"] == 512 else None)
        print(f"{cell['kf']:>6.0f} {winner:>8} {beta_h:>6.2f} {p_h:>6.2f} {rho_rf:>6.3f} {abs(pc[0]):>6.3f} "
              f"{A['cos']:>6.1f}/{A['resp']:>5.1f}/{A['chord']:>5.1f} {degmax:>9.3f}")
    # задача 3 сводка
    print("\n=== degen-гигиена ×512 (задача 3) ===")
    print("  degen-доли ×512 по θ (порог 0.2, что в сырье):",
          [round(d["cells"]["kf512.0"]["theta"][k]["degen"], 3) for k in d["cells"]["kf512.0"]["theta"]])
    print("  пер-репличные |proj| НЕ сохранены ⇒ вариация порога/исключение НЕВОЗМОЖНА из frozen raw.")
    print("  ⇒ амплитуды ×512 помечены НЕНАДЁЖНЫМИ; проекции — обязательны в C2-J.")
    json.dump(out, open(os.path.join(RES, "C2R_refit.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2R_refit.json")


if __name__ == "__main__":
    main()
