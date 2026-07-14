"""D-S3 анализ ПОСЛЕ коммита сырья. Карта анизотропии A(α); изотропизация
(Haar shared randomness R), фит −p·cosθ, изотропность, столбы {1/3, 2/π, 1/K_G},
прямой CHSH. Флип t̃=−t не меняет |E|.
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "fig")
SEED_R = 20260714          # фикс seed для Haar-R (репродуцируемая изотропизация)
THG = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) * np.pi / 8   # θ-сетка изотропизации
DEC = 0.2                  # порог DEGENERATE |n·axis|
PILLARS = {"1/3": 1/3, "2/pi": 2/np.pi, "1/K_G": 0.66}


def aicc(chi2, k, n):
    d = n - k - 1
    return chi2 + 2*k + (2*k*(k+1)/d if d > 0 else np.inf)


def axis(theta):
    return np.array([np.sin(theta), 0.0, np.cos(theta)])


def iso_corr(nA, nB, R, a, b):
    """Изотропизованная ⟨s·t⟩: s=sign(R nA·a), t=sign(R nB·b), среднее по ВСЕМ репликам.

    БЕЗ пост-селекции по |proj| (АУДИТ DS3: setting-зависимое DEGENERATE-отсечение —
    детекционный лупхол, фабриковал CHSH>2). Каждая реплика даёт ±1 (sign(0)→+1) ⇒
    λ=(R,nA,nB), s=s(λ,a), t=t(λ,b) — корректный LHV, |S|≤2 гарантирован. degen-доля
    (|proj|<DEC) возвращается лишь как ДИАГНОСТИКА, НЕ исключается."""
    RA = np.einsum("mij,mj->mi", R, nA)   # (M,3)
    RB = np.einsum("mij,mj->mi", R, nB)
    pa = RA @ a
    pb = RB @ b
    s = np.where(pa >= 0, 1, -1)
    t = np.where(pb >= 0, 1, -1)
    st = s * t
    E = float(np.mean(st))
    sig = float(np.sqrt(max(1 - E*E, 1e-9) / st.size))
    degen = float(np.mean((np.abs(pa) < DEC) | (np.abs(pb) < DEC)))
    return E, sig, degen


def main():
    d = json.load(open(os.path.join(RES, "DS3_raw.json")))
    out = {"aniso": {}, "iso": {}, "chsh": {}}

    # ===== 1. КАРТА АНИЗОТРОПИИ: A(α)=|E_anti| фит cosα/cos²α/exp =====
    print("=== DS3-H1 карта анизотропии A(α)=|E_anti| ===")
    al = np.array(d["meta"]["alpha_vals"])
    for mult in d["meta"]["KF"]:
        A = np.array([abs(d["aniso"][f"kf{mult}|a{an}|anti"]["E"]) for an in d["meta"]["alphas"]])
        s = np.array([d["aniso"][f"kf{mult}|a{an}|anti"]["sigma"] for an in d["meta"]["alphas"]])
        n = len(al)
        f_cos = lambda x, A0: A0*np.cos(x)
        f_cos2 = lambda x, A0: A0*np.cos(x)**2
        f_exp = lambda x, A0, w: A0*np.exp(-(x/w)**2)
        pc, _ = curve_fit(f_cos, al, A, p0=[A[0]], sigma=s, absolute_sigma=True)
        p2, _ = curve_fit(f_cos2, al, A, p0=[A[0]], sigma=s, absolute_sigma=True)
        pe, _ = curve_fit(f_exp, al, A, p0=[A[0], 1.0], sigma=s, absolute_sigma=True, maxfev=10000)
        c2 = {"cos": float(np.sum(((A-f_cos(al,*pc))/s)**2)),
              "cos2": float(np.sum(((A-f_cos2(al,*p2))/s)**2)),
              "exp": float(np.sum(((A-f_exp(al,*pe))/s)**2))}
        aa = {"cos": aicc(c2["cos"],1,n), "cos2": aicc(c2["cos2"],1,n), "exp": aicc(c2["exp"],2,n)}
        best = min(aa, key=aa.get)
        out["aniso"][f"kf{mult}"] = dict(alpha=list(al), A=list(map(float,A)), sigma=list(map(float,s)),
                                         chi2=c2, aicc=aa, best=best,
                                         params=dict(cos=float(pc[0]), cos2=float(p2[0]),
                                                     exp=[float(pe[0]),float(pe[1])]))
        print(f"  k_f×{mult}: A(α)={[f'{a:.3f}' for a in A]}  AICc cos={aa['cos']:.1f} "
              f"cos2={aa['cos2']:.1f} exp={aa['exp']:.1f} → {best}")

    # ===== 2. ИЗОТРОПИЗАЦИЯ =====
    print("=== DS3-H2 изотропизация (Haar shared randomness) ===")
    for mult in d["meta"]["KF"]:
        nA = np.array(d["source"][f"kf{mult}"]["n_A"])   # (M,3)
        nB = np.array(d["source"][f"kf{mult}"]["n_B"])
        Mr = nA.shape[0]
        R = Rotation.random(Mr, random_state=SEED_R + int(mult)).as_matrix()  # (M,3,3) Haar
        a0 = axis(0.0)
        E = np.zeros(len(THG)); sg = np.zeros(len(THG)); dg = np.zeros(len(THG))
        for i, th in enumerate(THG):
            E[i], sg[i], dg[i] = iso_corr(nA, nB, R, a0, axis(th))
        # фит −p·cosθ (флип: E→−E, т.е. E(θ)=+q cosθ, p=|q|)
        f = lambda x, q: q*np.cos(x)
        pq, _ = curve_fit(f, THG, E, p0=[E[0]], sigma=sg, absolute_sigma=True)
        p = abs(float(pq[0]))
        chi2 = float(np.sum(((E - f(THG,*pq))/sg)**2))
        # ТРЕУГОЛЬНИК ρ·(1−2θ/π) — LHV-корреляция (ожидается при n_A≈n_B, изотропизация)
        ftri = lambda x, rho: rho*(1 - 2*x/np.pi)
        pr, _ = curve_fit(ftri, THG, E, p0=[E[0]], sigma=sg, absolute_sigma=True)
        rho = abs(float(pr[0]))
        chi2_tri = float(np.sum(((E - ftri(THG,*pr))/sg)**2))
        shape = "cos" if chi2 < chi2_tri else "triangle(1−2θ/π)"
        # изотропность: повтор при повёрнутой лаб-паре (R0 фикс) на 3 θ
        R0 = Rotation.from_euler("xyz", [0.7, 1.1, 0.3]).as_matrix()
        iso_ok = True
        for th in [0.0, np.pi/4, np.pi/2]:
            E1, s1, _ = iso_corr(nA, nB, R, a0, axis(th))
            E2, s2, _ = iso_corr(nA, nB, R, R0 @ a0, R0 @ axis(th))
            if abs(E1 - E2) > 2*np.sqrt(s1**2 + s2**2):
                iso_ok = False
        out["iso"][f"kf{mult}"] = dict(theta=list(THG), E=list(map(float,E)), sigma=list(map(float,sg)),
                                       degen_diag=list(map(float,dg)), p_cos=p, chi2_cos=chi2,
                                       rho_triangle=rho, chi2_triangle=chi2_tri, best_shape=shape,
                                       isotropy_ok=bool(iso_ok))
        print(f"  k_f×{mult}: p_cos={p:.4f} (χ²={chi2:.0f})  ρ_triangle={rho:.4f} (χ²={chi2_tri:.0f}) "
              f"→ форма: {shape}  изотропность={iso_ok}")
        print(f"    столбы: 1/3={1/3:.3f}(Δ{p-1/3:+.3f}) 2/π={2/np.pi:.3f}(Δ{p-2/np.pi:+.3f}) "
              f"1/K_G=0.66(Δ{p-0.66:+.3f})  in[0.60,0.69]:{0.60<=p<=0.69}")

        # ===== 3. ПРЯМОЙ CHSH (изотропизованный, валиден) =====
        def Edir(aa, bb):
            e, ss, _ = iso_corr(nA, nB, R, aa, bb)
            return e, ss
        # a∈{0,π/2}, b∈{π/4,3π/4} — прямое измерение (изотропно ⇒ E=E(|a−b|))
        e00, s00 = Edir(axis(0.0), axis(np.pi/4))
        e01, s01 = Edir(axis(0.0), axis(3*np.pi/4))
        e10, s10 = Edir(axis(np.pi/2), axis(np.pi/4))
        e11, s11 = Edir(axis(np.pi/2), axis(3*np.pi/4))
        S = e00 - e01 + e10 + e11
        sS = np.sqrt(s00**2+s01**2+s10**2+s11**2)
        flag = "S≤2 ✓" if abs(S) <= 2+sS else "S>2 — АУДИТ!"
        out["chsh"][f"kf{mult}"] = dict(S=float(S), sigma=float(sS), ok=bool(abs(S)<=2+sS),
                                        E=dict(a0b0=e00,a0b1=e01,a1b0=e10,a1b1=e11))
        print(f"    прямой CHSH |S|={abs(S):.3f}±{sS:.3f}  {flag}")

    json.dump(out, open(os.path.join(RES, "DS3_analysis.json"), "w"), indent=2)

    # фигуры
    plt.figure(figsize=(7,5))
    for mult in d["meta"]["KF"]:
        r = out["aniso"][f"kf{mult}"]
        alf = np.linspace(0, np.pi/2, 100)
        plt.errorbar(al, r["A"], yerr=r["sigma"], fmt="o", capsize=3, label=f"k_f×{mult} (лучш. {r['best']})")
        plt.plot(alf, r["params"]["cos2"]*np.cos(alf)**2, "--", alpha=0.5)
    plt.xlabel("α (наклон оси a от ê)"); plt.ylabel("A(α)=|E_anti|")
    plt.title("DS3 карта анизотропии (--=cos²α)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ds3_aniso.png"), dpi=130); plt.close()

    plt.figure(figsize=(7.5,5))
    thf = np.linspace(0, np.pi, 100)
    for mult in d["meta"]["KF"]:
        r = out["iso"][f"kf{mult}"]
        line, = plt.plot([], [])
        col = line.get_color()
        plt.errorbar(THG, r["E"], yerr=r["sigma"], fmt="o", color=col, capsize=2,
                     label=f"k_f×{mult} (ρ_tri={r['rho_triangle']:.3f}, CHSH={2*r['rho_triangle']:.2f})")
        plt.plot(thf, r["rho_triangle"]*(1 - 2*thf/np.pi), "-", color=col, alpha=0.8)  # треугольник
        plt.plot(thf, r["p_cos"]*np.cos(thf), ":", color=col, alpha=0.4)  # косинус (плохой)
    for name, v in PILLARS.items():
        plt.axhline(v, ls=":", alpha=0.4)
        plt.text(0.05, v+0.01, name, fontsize=7)
    plt.xlabel("θ"); plt.ylabel("E_iso(θ)"); plt.title("DS3 изотропизованная E(θ)=p·cosθ + столбы")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ds3_iso.png"), dpi=130); plt.close()
    print(f"  анализ → {RES}/DS3_analysis.json ; фигуры → {FIG}/ds3_*.png")


if __name__ == "__main__":
    main()
