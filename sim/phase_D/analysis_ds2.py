"""D-S2 анализ ПОСЛЕ коммита сырья (DS2_prereg §порядок). Кросс-скан вердикт,
фиты формы (cos/tanh/sign-step, AICc), CHSH(k_f). Флип t̃=−t не меняет |E|.
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


def aicc(chi2, k, n):
    aic = chi2 + 2*k
    d = n - k - 1
    return aic + (2*k*(k+1)/d if d > 0 else np.inf)


def main():
    d = json.load(open(os.path.join(RES, "DS2_raw.json")))
    out = {"cross": {}, "form": {}, "chsh": {}}

    # ===== 1. КРОСС-СКАН: плато по N при k_f×1 и ×4? =====
    print("=== DS2-H1 кросс-скан A(N) ===")
    for mult in d["meta"]["kf_cross"]:
        Ns = d["meta"]["Ns_cross"]
        A = np.array([d["cross"][f"kf{mult}|N{N}"]["A"] for N in Ns])
        s = np.array([d["cross"][f"kf{mult}|N{N}"]["sigma"] for N in Ns])
        # монотонный спад за пределами 3σ между N=16 и N=96?
        drop = A[0] - A[-1]
        sig_drop = np.sqrt(s[0]**2 + s[-1]**2)
        monotone = all(A[i] >= A[i+1] for i in range(len(A)-1))
        decays = monotone and (drop > 3*sig_drop)
        out["cross"][f"kf{mult}"] = dict(N=Ns, A=list(map(float, A)), sigma=list(map(float, s)),
                                         drop_16_96=float(drop), sigma_drop=float(sig_drop),
                                         monotone_down=bool(monotone), decays_3sigma=bool(decays))
        print(f"  k_f×{mult}: A(N)={[f'{a:.3f}' for a in A]}  drop16→96={drop:+.3f}±{sig_drop:.3f}  "
              f"{'МОНОТ.СПАД>3σ (Kill?)' if decays else 'плато (нет спада>3σ)'}")
    kf_hi = f"kf{d['meta']['kf_cross'][-1]}"   # ключ старшей жёсткости (kf4.0)
    verdict_H1 = ("DS2-H1 ПОДТВЕРЖДЁН: плато при k_f×4 (нет спада>3σ) — механизм кинетический"
                  if not out["cross"][kf_hi]["decays_3sigma"]
                  else "KILL: A(k_f×4,N) спад>3σ — ξ конечна, плато D2-ext предасимптотично")
    out["verdict_H1"] = verdict_H1
    print(f"  ⇒ {verdict_H1}")

    # DS2-H3: k_f×1 воспроизводит плато D2-ext?
    ext = json.load(open(os.path.join(RES, "D2ext_analysis.json")))
    ext_A = dict(zip(ext["NS"], ext["A_N"]))
    print("  DS2-H3 (k_f×1 vs D2-ext):")
    for N in d["meta"]["Ns_cross"]:
        if N in ext_A:
            a1 = d["cross"][f"kf1.0|N{N}"]["A"]; s1 = d["cross"][f"kf1.0|N{N}"]["sigma"]
            print(f"    N{N}: DS2={a1:.3f}±{s1:.3f} vs D2-ext={ext_A[N]:.3f} (Δ={abs(a1-ext_A[N]):.3f})")

    # ===== 2. ФОРМА: фиты cos/tanh/sign-step =====
    print("=== DS2-H2 форма (фиты, флип E→−E) ===")
    def f_cos(x, A): return -A*np.cos(x)
    def f_tanh(x, beta): return -np.tanh(beta*np.cos(x))
    def f_sign(x, A): return -A*np.sign(np.cos(x))
    betas = {}
    form_fits = {}
    for mult in d["meta"]["kf_form"]:
        th = np.array([THETA_MAP[n] for n in d["meta"]["thetas"]])
        E = np.array([-d["form"][f"kf{mult}|{n}"]["E_raw"] for n in d["meta"]["thetas"]])  # флип
        sg = np.array([d["form"][f"kf{mult}|{n}"]["sigma"] for n in d["meta"]["thetas"]])
        n = len(th)
        pc, _ = curve_fit(f_cos, th, E, p0=[0.5], sigma=sg, absolute_sigma=True)
        pt, _ = curve_fit(f_tanh, th, E, p0=[0.5], sigma=sg, absolute_sigma=True, maxfev=10000)
        ps, _ = curve_fit(f_sign, th, E, p0=[0.5], sigma=sg, absolute_sigma=True)
        c2c = float(np.sum(((E-f_cos(th,*pc))/sg)**2))
        c2t = float(np.sum(((E-f_tanh(th,*pt))/sg)**2))
        c2s = float(np.sum(((E-f_sign(th,*ps))/sg)**2))
        A = {"cos": aicc(c2c,1,n), "tanh": aicc(c2t,1,n), "sign": aicc(c2s,1,n)}
        best = min(A, key=A.get)
        betas[mult] = float(pt[0])
        form_fits[f"kf{mult}"] = dict(A_cos=float(pc[0]), beta=float(pt[0]), amp_tanh=float(np.tanh(pt[0])),
                                      A_sign=float(ps[0]), chi2=dict(cos=c2c,tanh=c2t,sign=c2s),
                                      aicc=A, best=best, dAICc={m: float(A[m]-A[best]) for m in A})
        print(f"  k_f×{mult}: β={pt[0]:.3f} (amp={np.tanh(pt[0]):.3f}) | AICc cos={A['cos']:.1f} "
              f"tanh={A['tanh']:.1f} sign={A['sign']:.1f} → лучшая {best} (Δcos={A['cos']-A[best]:.1f})")
    out["form"] = form_fits
    out["beta_trend"] = {f"kf{m}": betas[m] for m in betas}
    trend_up = all(betas[d["meta"]["kf_form"][i+1]] > betas[d["meta"]["kf_form"][i]]
                   for i in range(len(d["meta"]["kf_form"])-1))
    print(f"  β(k_f)={[round(betas[m],3) for m in d['meta']['kf_form']]}  монотонный рост: {trend_up}")

    # ===== 3. CHSH(k_f) =====
    print("=== DS2-H2 CHSH(k_f) ===")
    def Eth(mult, thn, flip=True):
        c = d["form"][f"kf{mult}|{thn}"]
        return (-1.0 if flip else 1.0)*c["E_raw"], c["sigma"]
    for mult in d["meta"]["kf_form"]:
        e14, s14 = Eth(mult, "pi/4"); e34, s34 = Eth(mult, "3pi/4")
        S = 3*e14 - e34; sS = np.sqrt((3*s14)**2 + s34**2)
        # что дал бы чистый косинус при этой амплитуде A:
        A0 = form_fits[f"kf{mult}"]["A_cos"]
        S_cos = A0*(3*np.cos(np.pi/4) - np.cos(3*np.pi/4))  # = A0*2.828
        flag = "S≤2 ✓" if abs(S) <= 2.0+sS else "S>2 — АУДИТ!"
        out["chsh"][f"kf{mult}"] = dict(S=float(S), sigma=float(sS), S_cos_would_be=float(S_cos), ok=bool(abs(S)<=2.0+sS))
        print(f"  k_f×{mult}: |S|={abs(S):.3f}±{sS:.3f}  (косинус дал бы {S_cos:.3f})  {flag}")

    json.dump(out, open(os.path.join(RES, "DS2_analysis.json"), "w"), indent=2)

    # ===== фигуры =====
    plt.figure(figsize=(7,5))
    for mult in d["meta"]["kf_cross"]:
        Ns = d["meta"]["Ns_cross"]
        A = [d["cross"][f"kf{mult}|N{N}"]["A"] for N in Ns]
        s = [d["cross"][f"kf{mult}|N{N}"]["sigma"] for N in Ns]
        plt.errorbar(Ns, A, yerr=s, fmt="o-", capsize=3, label=f"k_f×{mult}")
    plt.xlabel("N"); plt.ylabel("A(N)"); plt.ylim(0,1)
    plt.title("DS2 cross-scan: A(N) at two twist stiffnesses")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ds2_cross.png"), dpi=130); plt.close()

    plt.figure(figsize=(7.5,5))
    thf = np.linspace(0, np.pi, 200)
    for i, mult in enumerate(d["meta"]["kf_form"]):
        th = np.array([THETA_MAP[n] for n in d["meta"]["thetas"]])
        E = np.array([-d["form"][f"kf{mult}|{n}"]["E_raw"] for n in d["meta"]["thetas"]])
        sg = np.array([d["form"][f"kf{mult}|{n}"]["sigma"] for n in d["meta"]["thetas"]])
        col = f"C{i}"
        plt.errorbar(th, E, yerr=sg, fmt="o", color=col, capsize=2, label=f"k_f×{mult} (β={betas[mult]:.2f})")
        plt.plot(thf, -np.tanh(betas[mult]*np.cos(thf)), "-", color=col, alpha=0.8)
    plt.plot(thf, -np.cos(thf), "k:", alpha=0.4, label="−cosθ (pure)")
    plt.xlabel("θ"); plt.ylabel("E(θ) [flip]"); plt.title("DS2 shape: E(θ) vs k_f (—=tanh fit)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "ds2_form.png"), dpi=130); plt.close()
    print(f"  анализ → {RES}/DS2_analysis.json ; фигуры → {FIG}/ds2_*.png")


if __name__ == "__main__":
    main()
