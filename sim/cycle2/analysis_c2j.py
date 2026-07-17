"""C2-J лестница — анализ ПОСЛЕ коммита сырья. Траектория по k_c; 3-модельный
форм-фит (косинус / семья-отклика норм. / хорда); degen-гигиена (проекции при
θ=0); H-J1..J5 + рельс {F_s=0,S>2}. Код до сырья.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")


def sphere_nodes(n):
    ct = np.linspace(-1, 1, n); phi = np.linspace(0, 2*np.pi, n, endpoint=False)
    CT, PH = np.meshgrid(ct, phi); ST = np.sqrt(np.clip(1-CT**2, 0, 1))
    return np.stack([ST*np.cos(PH), ST*np.sin(PH), CT], -1).reshape(-1, 3)


NODES = sphere_nodes(64)


def Ehat_beta(theta_arr, beta):
    tb = np.tanh(beta) if beta > 1e-8 else beta
    def C_(th):
        a = np.array([0, 0, 1.0]); b = np.array([np.sin(th), 0, np.cos(th)])
        ga = np.tanh(beta*(NODES@a))/tb; gb = np.tanh(beta*(NODES@b))/tb
        return float(np.mean(ga*gb))
    c0 = C_(0.0)
    return np.array([C_(t)/c0 for t in np.atleast_1d(theta_arr)])


def aicc(chi2, k, n):
    dd = n-k-1
    return chi2 + 2*k + (2*k*(k+1)/dd if dd > 0 else np.inf)


def fit3(th, E, sg):
    n = len(th)
    fcos = lambda x, r: r*np.cos(x)
    pc, _ = curve_fit(fcos, th, E, p0=[E[0]], sigma=sg, absolute_sigma=True)
    chi_c = float(np.sum(((E-fcos(th,*pc))/sg)**2))
    try:
        prf, _ = curve_fit(lambda x, r, b: r*Ehat_beta(x, b), th, E, p0=[E[0], 1.0],
                           sigma=sg, absolute_sigma=True, bounds=([-2, 0.02], [2, 30]), maxfev=6000)
        chi_rf = float(np.sum(((E-prf[0]*Ehat_beta(th, prf[1]))/sg)**2)); beta_h, rho_rf = float(prf[1]), float(prf[0])
    except Exception:
        chi_rf, beta_h, rho_rf = np.inf, np.nan, np.nan
    def Ep(x, p):
        s = np.sin(x/2)**p; c = np.cos(x/2)**p; return (s-c)/(s+c)
    pch, _ = curve_fit(lambda x, r, p: -r*Ep(x, p), th, E, p0=[abs(E[0]), 2.0],
                       sigma=sg, absolute_sigma=True, bounds=([0, 0.3], [2, 8]), maxfev=20000)
    chi_ch = float(np.sum(((E+pch[0]*Ep(th, pch[1]))/sg)**2)); p_h = float(pch[1])
    A = {"cos": aicc(chi_c,1,n), "resp": aicc(chi_rf,2,n), "chord": aicc(chi_ch,2,n)}
    return dict(winner=min(A, key=A.get), beta_hat=beta_h, p_hat=p_h, rho_rf=rho_rf,
                rho_cos=abs(float(pc[0])), aicc=A)


def degen_hygiene(s0, t0, pa0, pb0):
    """E(0) при порогах {0.1,0.2,0.3} (все события, sign(0)→+1) и при ИСКЛЮЧЕНИИ degen."""
    s0, t0, pa0, pb0 = map(np.array, (s0, t0, pa0, pb0))
    st = s0*t0
    res = {}
    E_all = float(np.mean(st))
    for thr in (0.1, 0.2, 0.3):
        deg = (pa0 < thr) | (pb0 < thr)
        keep = ~deg
        E_excl = float(np.mean(st[keep])) if keep.sum() else 0.0
        res[f"thr{thr}"] = dict(E_all=E_all, E_excl_degen=E_excl, degen_frac=float(np.mean(deg)))
    # дрейф E_all vs E_excl при 0.2
    d02 = res["thr0.2"]
    drift = abs(d02["E_all"] - d02["E_excl_degen"])
    sig = np.sqrt(max(1-E_all**2, 1e-9)/len(st))
    return res, float(drift), float(drift/sig)


def main():
    d = json.load(open(os.path.join(RES, "C2J_ladder_raw.json")))
    out = {"meta": dict(raw="C2J_ladder_raw.json", prereg=d["meta"]["prereg_commit"]), "trajectory": []}
    print("=== C2-J траектория по k_c ===")
    print(f"{'k_c':>6} {'F_s32':>8} {'F_s8':>8} {'Ft32':>7} {'форма':>7} {'β̂':>6} {'p̂':>6} "
          f"{'ρ':>6} {'|E0|':>6} {'|S|':>7} {'C_ee':>6} {'deg':>5} {'E0drift':>8}")
    kink8 = None; rail_hit = None
    for ck, cell in d["cells"].items():
        kc = cell["kc"]
        Fs32, sF32 = cell["F_s"]["N32"]["F_s"], cell["F_s"]["N32"]["sigma"]
        Fs8, sF8 = cell["F_s"]["N8"]["F_s"], cell["F_s"]["N8"]["sigma"]
        th = np.array([cell["theta"][k]["theta"] for k in cell["theta"]])
        E = np.array([cell["theta"][k]["E"] for k in cell["theta"]])
        sg = np.array([cell["theta"][k]["sigma"] for k in cell["theta"]])
        ff = fit3(th, E, sg)
        S, sS = cell["chsh"]["S"], cell["chsh"]["sigma"]
        r = cell["response"]
        hyg, drift, drift_sig = degen_hygiene(r["s_theta0"], r["t_theta0"], r["proj_A0"], r["proj_B0"])
        E0 = cell["theta"]["0.0000"]["E"]; degmax = max(cell["theta"][k]["degen"] for k in cell["theta"])
        e0_unreliable = drift_sig > 2
        if kink8 is None and Fs8 > 3*max(sF8, 1e-9) and Fs8 > 0:
            kink8 = kc
        rail = cell.get("rail_violation") or ((Fs32 < 3*max(sF32,1e-9)) and abs(S) > 2+sS)
        if rail and rail_hit is None:
            rail_hit = kc
        out["trajectory"].append(dict(kc=kc, F_s_N32=Fs32, F_s_N8=Fs8, F_t_N32=cell["F_s"]["N32"]["F_t"],
            Delta_N32=cell["F_s"]["N32"]["Delta"], form_winner=ff["winner"], beta_hat=ff["beta_hat"],
            p_hat=ff["p_hat"], rho_rf=ff["rho_rf"], rho_cos=ff["rho_cos"], E0=E0, S=S, sigma_S=sS,
            C_ee=cell["C_ee"], degen_max=degmax, E0_drift_sigma=drift_sig, E0_unreliable=bool(e0_unreliable),
            aicc=ff["aicc"], S_gt_2=bool(abs(S) > 2+sS), rail=bool(rail)))
        flag = " ⚠E0" if e0_unreliable else ""
        print(f"{kc:>6.0f} {Fs32:>8.4f} {Fs8:>8.4f} {cell['F_s']['N32']['F_t']:>7.3f} {ff['winner']:>7} "
              f"{ff['beta_hat']:>6.2f} {ff['p_hat']:>6.2f} {ff['rho_cos']:>6.3f} {abs(E0):>6.3f} "
              f"{abs(S):>7.4f} {cell['C_ee']:>6.3f} {degmax:>5.2f} {drift_sig:>7.1f}σ{flag}")
        if rail:
            print(f"    ⚠⚠ РЕЛЬС: F_s≈0 И |S|={abs(S):.3f}>2 — БЕЗУСЛОВНЫЙ СТОП-АУДИТ")

    print("\n=== вердикты ===")
    if kink8 is None:
        print("H-J1: KILL — F_s≈0 до k_c×256 обе N; совместный режим по цепной ручке недостижим в диапазоне.")
    else:
        print(f"H-J1: излом F_s при k_c×{kink8:.0f} (N=8).")
    out["H_J1_kink_kc"] = kink8; out["rail_hit_kc"] = rail_hit
    json.dump(out, open(os.path.join(RES, "C2J_ladder_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2J_ladder_analysis.json")


if __name__ == "__main__":
    main()
