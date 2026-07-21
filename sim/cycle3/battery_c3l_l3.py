"""C3-L / L3 батарея (prereg 2d78c51). Ячейка1: скан Δ(p,χ=0.5),
p∈{4,6,10,∞}. Ячейка2: внутренний мост S(p), p∈{1,1.5,2,3,4,∞}.
numpy, N=2e6, аналитика Йенсен-зазора рядом. --smoke для дыма.
"""
import json
import os
import sys

import numpy as np

SMOKE = "--smoke" in sys.argv
N = 200_000 if SMOKE else 2_000_000
RNG = np.random.default_rng(20260721)
CHI = 0.5
SCAN_P = [4.0, 6.0, 10.0, np.inf]                     # ячейка 1
BRIDGE_P = [1.0, 1.5, 2.0, 3.0, 4.0, np.inf]          # ячейка 2 (A7)
Z = np.array([0.0, 0.0, 1.0]); X = np.array([1.0, 0.0, 0.0])
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")


def f_p(c, p):
    """Правило считывания; p=∞ = ступень 1{c>0} (c=0 → 1/2)."""
    if np.isinf(p):
        return np.where(c > 0, 1.0, np.where(c < 0, 0.0, 0.5))
    A = np.abs(1 + c) ** (p / 2); B = np.abs(1 - c) ** (p / 2)
    return A / (A + B)


def sample_source(n=N):
    """λ~S² с плотностью 1+χλ_z (та же мера, что C3-B F2)."""
    mx = 1 + CHI * 1.0; out = []; need = n
    while need > 0:
        v = RNG.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < (1 + CHI * v[:, 2]) / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def cond_means(lam, a):
    """Условные средние m_± и веса P_± при настройке Алисы a (s=sign(λ·a))."""
    proj = lam @ a; pos = proj >= 0
    Pp = pos.mean(); Pm = 1 - Pp
    mp = lam[pos].mean(0); mm = lam[~pos].mean(0)
    return Pp, Pm, mp, mm


def bobP(lam, a, b, p):
    """P_Bob(+) при настройке Алисы a, чтении Боба b (маргинал по s)."""
    Pp, Pm, mp, mm = cond_means(lam, a)
    return Pp * f_p(mp @ b, p) + Pm * f_p(mm @ b, p)


def steer_gap(lam, p):
    """Δ(p) = |P_Bob(+|a=ẑ) − P_Bob(+|a=x̂)|, Боб читает ẑ."""
    return abs(bobP(lam, Z, Z, p) - bobP(lam, X, Z, p))


# ---- аналитика Йенсен-зазора (1D по c=λ_z, мера 1+χc равномерна в cosθ) ----
def analytic_gap(p, ngrid=200001):
    """Δ_ан(p): a=x̂ тривиально (симметрия по x ⇒ P_Bob=f_p(0)=1/2);
    a=ẑ даёт зазор E[f_p]−f_p(0). f_∞: точная кусочная (доли c>0/<0)."""
    c = np.linspace(-1, 1, ngrid); g = 1 + CHI * c; g = g / np.trapezoid(g, c)
    pos = c >= 0
    Pp = np.trapezoid(g[pos], c[pos]); Pm = np.trapezoid(g[~pos], c[~pos])
    mp = np.trapezoid(c[pos] * g[pos], c[pos]) / Pp
    mm = np.trapezoid(c[~pos] * g[~pos], c[~pos]) / Pm
    bob_z = Pp * f_p(np.array([mp]), p)[0] + Pm * f_p(np.array([mm]), p)[0]
    bob_x = 0.5   # a=x̂: m_± симметричны по z ⇒ m_±·ẑ=±0, но раздельно:
    # строго: при a=x̂ под-ансамбли {λ_x><0} имеют ⟨λ_z⟩ одинаков (=lz),
    # ⇒ bob_x = f_p(lz). lz:
    lz = np.trapezoid(c * g, c)
    bob_x = f_p(np.array([lz]), p)[0]
    return abs(bob_z - bob_x)


def cell1_scan():
    print("=" * 72); print("Ячейка 1 — скан Δ(p,χ=0.5), p∈{4,6,10,∞}"); print("=" * 72)
    lam = sample_source()
    out = {}; prev = -1; mono = True
    print(f"  {'p':>5} {'Δ числ':>12} {'Δ аналит':>12} {'|ан−чис|':>10} {'<2σ':>6}")
    for p in SCAN_P:
        dn = steer_gap(lam, p); da = analytic_gap(p)
        mism = abs(dn - da); ok = mism < 2 * SIG
        key = "inf" if np.isinf(p) else f"{p}"
        out[key] = dict(num=float(dn), analytic=float(da), mismatch=float(mism), match_2sig=bool(ok))
        if dn + 1e-12 < prev - 2 * SIG:
            mono = False
        prev = dn
        print(f"  {key:>5} {dn:>12.6f} {da:>12.6f} {mism:>10.2e} {str(ok):>6}")
        if not ok:
            print(f"    ⚠⚠ СТОП-КАНДИДАТ: |аналит−числ|>2σ={2*SIG:.4f} при p={key}")
    dmax_at_inf = max(out, key=lambda k: out[k]["num"]) == "inf"
    print(f"  ВЕРДИКТ: монотонность {mono}, Δ(∞) максимален {dmax_at_inf}, "
          f"все Δ>0 {all(v['num'] > 5*SIG for v in out.values())}")
    return dict(cells=out, monotone=bool(mono), max_at_inf=bool(dmax_at_inf),
                all_positive=bool(all(v['num'] > 5*SIG for v in out.values())))


def cell2_bridge():
    print("\n" + "=" * 72); print("Ячейка 2 — внутренний мост S(p), p∈{1,1.5,2,3,4,∞}"); print("=" * 72)
    lam = sample_source()

    def setting(deg):
        t = np.radians(deg); return np.array([np.sin(t), 0.0, np.cos(t)])
    a, ap, b, bp = setting(0), setting(90), setting(45), setting(135)

    def E(av, bv, p):
        Pp, Pm, mp, mm = cond_means(lam, av)
        # s=+1 вес Pp, ⟨t|+⟩=2f−1; s=−1 вес Pm, ⟨t|−⟩=2f−1
        return Pp * (+1) * (2 * f_p(mp @ bv, p) - 1) + Pm * (-1) * (2 * f_p(mm @ bv, p) - 1)

    out = {}; prev = -1; mono = True
    print(f"  {'p':>5} {'E(a,b)':>9} {'E(a,b′)':>9} {'E(a′,b)':>9} {'E(a′,b′)':>9} {'S':>9}")
    for p in BRIDGE_P:
        e1, e2, e3, e4 = E(a, b, p), E(a, bp, p), E(ap, b, p), E(ap, bp, p)
        S = abs(e1 - e2 + e3 + e4)
        key = "inf" if np.isinf(p) else f"{p}"
        out[key] = dict(E_ab=float(e1), E_abp=float(e2), E_apb=float(e3), E_apbp=float(e4), S=float(S))
        if S + 1e-9 < prev - 2 * SIG:
            mono = False
        prev = S
        print(f"  {key:>5} {e1:>9.4f} {e2:>9.4f} {e3:>9.4f} {e4:>9.4f} {S:>9.4f}")
    s2 = out["2.0"]["S"]; tsirelson = 2 * np.sqrt(2)
    near = abs(s2 - tsirelson) < 3 * SIG
    print(f"  S(2)={s2:.4f} vs 2√2={tsirelson:.4f} (|Δ|={abs(s2-tsirelson):.4f}, "
          f"3σ={3*SIG:.4f}: {'≈' if near else '≠'})")
    print(f"  ВЕРДИКТ: монотонность S(p) {mono}; S(2)≈2√2 {near}")
    return dict(cells=out, monotone=bool(mono), S2=float(s2), tsirelson=float(tsirelson),
                S2_near_tsirelson=bool(near))


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N}, χ={CHI}, σ={SIG:.2e}\n")
    c1 = cell1_scan(); c2 = cell2_bridge()
    tag = "_smoke" if SMOKE else ""
    path = os.path.join(RES, f"C3L_L3{tag}.json")
    json.dump(dict(cell1_scan=c1, cell2_bridge=c2, sigma=SIG, N=N, chi=CHI,
                   scan_p=[("inf" if np.isinf(p) else p) for p in SCAN_P],
                   bridge_p=[("inf" if np.isinf(p) else p) for p in BRIDGE_P]),
              open(path, "w"), indent=2)
    print(f"\n  → {path}")


if __name__ == "__main__":
    main()
