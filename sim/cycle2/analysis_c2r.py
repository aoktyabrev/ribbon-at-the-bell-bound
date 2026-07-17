"""C2-R лестница — анализ ПОСЛЕ коммита сырья. H-R1..R6 + траектория.
Три модели формы (косинус / семья отклика tanh(βc) / хорда), AICc;
функция отклика P(s=+|c) (H-R6). Код до сырья.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")
K_G_INV = 0.66            # 1/K_G(3) столб (H-R4b)


def aicc(chi2, k, n):
    d = n - k - 1
    return chi2 + 2*k + (2*k*(k+1)/d if d > 0 else np.inf)


def resp_family(theta, rho, beta):
    """E_β(θ)=−∫ g(λ·a)g(λ·b)dλ/4π, g(c)=tanh(βc)/tanh(β); квадратура по S²."""
    # a=ê, b=axis(θ); интеграл по направлениям λ на сфере
    nphi, nct = 48, 48
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    ct = np.linspace(-1, 1, nct)
    st = np.sqrt(1 - ct**2)
    PH, CT = np.meshgrid(phi, ct)
    ST = np.sqrt(1 - CT**2)
    lx, ly, lz = ST*np.cos(PH), ST*np.sin(PH), CT
    out = np.zeros_like(theta, dtype=float)
    tb = np.tanh(beta) if beta > 1e-6 else beta
    for i, th in enumerate(np.atleast_1d(theta)):
        a = np.array([0, 0, 1.0]); b = np.array([np.sin(th), 0, np.cos(th)])
        ca = lx*a[0]+ly*a[1]+lz*a[2]; cb = lx*b[0]+ly*b[1]+lz*b[2]
        ga = np.tanh(beta*ca)/tb; gb = np.tanh(beta*cb)/tb
        integ = np.mean(ga*gb)     # среднее по равномерной сфере = ∫/4π
        out[i] = -rho*integ
    return out if out.size > 1 else out[0]


def fit_forms(th, E, sg):
    n = len(th)
    fcos = lambda x, r: -r*np.cos(x)
    pc, _ = curve_fit(fcos, th, E, p0=[abs(E[0])], sigma=sg, absolute_sigma=True)
    chi_c = float(np.sum(((E-fcos(th,*pc))/sg)**2))
    # хорда
    def Ep(x, p):
        s = np.sin(x/2)**p; c = np.cos(x/2)**p; return (s-c)/(s+c)
    fch = lambda x, r, p: -r*Ep(x, p)
    try:
        pch, _ = curve_fit(fch, th, E, p0=[abs(E[0]), 2.0], sigma=sg,
                           absolute_sigma=True, bounds=([0,0.3],[2,8]), maxfev=20000)
        chi_ch = float(np.sum(((E-fch(th,*pch))/sg)**2)); p_h = float(pch[1]); r_ch = float(pch[0])
    except Exception:
        chi_ch, p_h, r_ch = np.inf, np.nan, np.nan
    # семья отклика
    frf = lambda x, r, b: resp_family(x, r, b)
    try:
        prf, _ = curve_fit(frf, th, E, p0=[abs(E[0])*3, 1.0], sigma=sg,
                           absolute_sigma=True, bounds=([0,0.05],[5,20]), maxfev=8000)
        chi_rf = float(np.sum(((E-frf(th,*prf))/sg)**2)); beta_h = float(prf[1]); r_rf = float(prf[0])
    except Exception:
        chi_rf, beta_h, r_rf = np.inf, np.nan, np.nan
    A = {"cos": aicc(chi_c,1,n), "resp": aicc(chi_rf,2,n), "chord": aicc(chi_ch,2,n)}
    winner = min(A, key=A.get)
    return dict(rho_cos=abs(float(pc[0])), chi_cos=chi_c, p_hat=p_h, chi_chord=chi_ch,
                beta_hat=beta_h, rho_rf=r_rf, chi_resp=chi_rf, aicc=A, winner=winner)


def response_diag(c, s):
    """P(s=+|c) бины 0.1; фит (1+κc)/2 vs знаковая ступень."""
    c = np.asarray(c); s = np.asarray(s)
    edges = np.arange(-1, 1.001, 0.1); mid = 0.5*(edges[:-1]+edges[1:])
    P = []; nb = []
    for i in range(len(edges)-1):
        m = (c >= edges[i]) & (c < edges[i+1]); n = int(m.sum())
        P.append(float(np.mean(s[m] > 0)) if n else np.nan); nb.append(n)
    P = np.array(P); mid_ok = ~np.isnan(P)
    flin = lambda x, k: 0.5*(1+k*x)
    try:
        pk, _ = curve_fit(flin, mid[mid_ok], P[mid_ok], p0=[1.0])
        kappa = float(pk[0])
        chi_lin = float(np.sum((P[mid_ok]-flin(mid[mid_ok],*pk))**2))
    except Exception:
        kappa, chi_lin = np.nan, np.inf
    # знаковая ступень P=1 при c>0
    step = np.where(mid[mid_ok] > 0, 1.0, 0.0)
    chi_step = float(np.sum((P[mid_ok]-step)**2))
    return dict(bins=list(map(float, mid)), P=list(map(lambda x: None if np.isnan(x) else float(x), P)),
                n=nb, kappa=kappa, chi_lin=chi_lin, chi_step=chi_step,
                shape="линейная" if chi_lin < chi_step else "ступень")


def main():
    d = json.load(open(os.path.join(RES, "C2R_ladder_raw.json")))
    out = {"meta": dict(raw="C2R_ladder_raw.json", prereg=d["meta"]["prereg_commit"]),
           "trajectory": []}
    print("=== C2-R траектория (k_f, форма, β̂/p̂, ρ, F_s, Δ, S, κ̂) ===")
    print(f"{'k_f':>6} {'F_s(32)':>9} {'F_s(8)':>9} {'форма':>7} {'β̂':>6} {'p̂':>6} {'ρ_cos':>6} {'|S|':>6} {'κ̂':>6} {'ρ=κ/3':>7}")
    kink32 = None
    for ck, cell in d["cells"].items():
        kf = cell["kf"]
        Fs32 = cell["F_s"]["N32"]["F_s"]; sF32 = cell["F_s"]["N32"]["sigma"]
        Fs8 = cell["F_s"]["N8"]["F_s"]; sF8 = cell["F_s"]["N8"]["sigma"]
        th = np.array([cell["theta"][k]["theta"] for k in cell["theta"]])
        E = np.array([cell["theta"][k]["E"] for k in cell["theta"]])
        sg = np.array([cell["theta"][k]["sigma"] for k in cell["theta"]])
        ff = fit_forms(th, E, sg)
        S = cell["chsh"]["S"]; sS = cell["chsh"]["sigma"]
        rd = response_diag(cell["response"]["c"], cell["response"]["s_theta0"])
        # H-R1 kink (N=32): F_s>3σ_F
        if kink32 is None and Fs32 > 3*max(sF32, 1e-9) and Fs32 > 0:
            kink32 = kf
        # H-R4b столб
        pillar_viol = (ff["rho_cos"] > K_G_INV) and (Fs32 < 3*max(sF32, 1e-9))
        rho_from_kappa = rd["kappa"]/3 if not np.isnan(rd["kappa"]) else np.nan
        row = dict(kf=kf, F_s_N32=Fs32, sigma_N32=sF32, F_s_N8=Fs8, Delta_N32=cell["F_s"]["N32"]["Delta"],
                   form_winner=ff["winner"], beta_hat=ff["beta_hat"], p_hat=ff["p_hat"],
                   rho_cos=ff["rho_cos"], S=S, sigma_S=sS, aicc=ff["aicc"],
                   kappa=rd["kappa"], rho_from_kappa=rho_from_kappa, response_shape=rd["shape"],
                   pillar_violation=bool(pillar_viol),
                   S_gt_2=bool(abs(S) > 2 + sS))
        out["trajectory"].append(row)
        print(f"{kf:>6.0f} {Fs32:>9.4f} {Fs8:>9.4f} {ff['winner']:>7} {ff['beta_hat']:>6.2f} "
              f"{ff['p_hat']:>6.2f} {ff['rho_cos']:>6.3f} {abs(S):>6.3f} {rd['kappa']:>6.3f} {rho_from_kappa:>7.3f}")
        if pillar_viol:
            print(f"    ⚠ H-R4b СТОЛБ НАРУШЕН: ρ_cos={ff['rho_cos']:.3f}>0.66 при F_s≈0 — Гротендик-стоп-аудит")
        if abs(S) > 2 + sS:
            print(f"    ⚠ S>2 ({abs(S):.3f}): H-R3 СТОП-АУДИТ до интерпретации")

    # вердикты
    print("\n=== вердикты ===")
    if kink32 is None:
        print("H-R1: KILL — F_s<3σ на всей лестнице до ×512 (N=32); факторизация устойчива, "
              "граница за окном (PR-предел только в строгом κ→∞).")
    else:
        print(f"H-R1: излом при k_f×{kink32:.0f} (N=32). H-R2: сравнить порог N=8 vs N=32 в raw.")
    out["H_R1_kink_kf"] = kink32
    json.dump(out, open(os.path.join(RES, "C2R_ladder_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2R_ladder_analysis.json")


if __name__ == "__main__":
    main()
