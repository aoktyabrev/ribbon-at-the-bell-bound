"""C3-B упрочнение (addendum1). T1 цепные/порядок; T2 анти-циркулярность
(деформации меры, робастность нуль-множества {2}); T3 F1. numpy N=2e6, A4-сетки.
"""
import json
import os

import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260720)
P_GRID = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
Z = np.array([0.0, 0.0, 1.0]); X = np.array([1.0, 0.0, 0.0])
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")


def f_p(c, p):
    A = np.abs(1 + c) ** (p / 2); B = np.abs(1 - c) ** (p / 2)
    return A / (A + B)


def sample_density(dens_c, n=N):
    """λ~S² с плотностью dens_c(λ_z) (нормируется макс на [-1,1])."""
    cc = np.linspace(-1, 1, 2001); mx = dens_c(cc).max()
    out = []; need = n
    while need > 0:
        v = RNG.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < dens_c(v[:, 2]) / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def steer_scan(lam, p):
    """Δ(p) = |P(Bob+|Alice ẑ) − P(Bob+|Alice x̂)|, Боб чит. ẑ (сырое усл. среднее)."""
    def bobP(a):
        proj = lam @ a; pos = proj >= 0; P = 0.0
        for mask in (pos, ~pos):
            w = mask.mean(); nb = lam[mask].mean(0); P += w * f_p(nb @ Z, p)
        return P
    return abs(bobP(Z) - bobP(X))


def D_moment(lam):
    """D = |3-й момент z-разложения a=ẑ − a=x̂| при равном среднем."""
    def mom(a, k):
        proj = lam @ a; pos = proj >= 0; m = 0.0
        for mask in (pos, ~pos):
            w = mask.mean(); nb = lam[mask].mean(0); m += w * (nb @ Z) ** k
        return m
    if abs(mom(Z, 1) - mom(X, 1)) > 5 * SIG:
        return np.nan
    return abs(mom(Z, 3) - mom(X, 3))


def zero_set(lam):
    return [p for p in P_GRID if steer_scan(lam, p) < 2 * SIG]


def analytic_Delta(dens_c, p):
    """Точная Δ(p) = |P₊f_p(m₊)+P₋f_p(m₋) − f_p(⟨λ_z⟩)| (a=ẑ vs a=x̂ тривиально),
    1D-интеграл по c=λ_z с плотностью dens_c (равномерная мера в cosθ).
    addendum2: в p=2 — АЛГЕБРАИЧЕСКИЙ ноль, не квадратура. f₂(c)=(1+c)/2 аффинна
    ⇒ P₊f₂(m₊)+P₋f₂(m₋) = ½[(P₊+P₋)+(P₊m₊+P₋m₋)] = ½[1+⟨λ_z⟩] = f₂(⟨λ_z⟩)
    тождественно (зазор Йенсена аффинной функции ≡ 0) ∀ меры ⇒ Δ(2)≡0 точно.
    Квадратура здесь давала ошибку трапеции ~2.3e-5 (1/ngrid) → ложный СТОП."""
    if abs(p - 2.0) < 1e-12:
        return 0.0
    c = np.linspace(-1, 1, 20001); g = dens_c(c); g = g / np.trapezoid(g, c)
    pos = c >= 0
    Pp = np.trapezoid(g[pos], c[pos]); Pm = np.trapezoid(g[~pos], c[~pos])
    mp = np.trapezoid(c[pos]*g[pos], c[pos]) / Pp
    mm = np.trapezoid(c[~pos]*g[~pos], c[~pos]) / Pm
    lz = np.trapezoid(c*g, c)
    return abs(Pp*f_p(mp, p) + Pm*f_p(mm, p) - f_p(lz, p))


def analytic_zero_set(dens_c, tol=1e-9):
    return [p for p in P_GRID if analytic_Delta(dens_c, p) < tol]


def T1_chain():
    print("="*72); print("T1 — цепные/порядок (B7) на F2 (χ=0.5)"); print("="*72)
    lam = sample_density(lambda c: 1 + 0.5 * c)
    # (i) повторяемость: измерить b, ограничить, пере-измерить b → тот же t
    b = Z; t1 = np.sign(lam @ b); sub = lam[t1 > 0]
    t2 = np.sign(sub @ b); rep = float(np.mean(t2 > 0))
    print(f"  (i) повторяемость: пере-измерение b на {{sign(b·λ)>0}} → P(+)={rep:.4f} (=1.0 ⇒ репитабельно)")
    # (ii) порядок: {a}∩{b} = {b}∩{a} (пересечение коммутирует)
    a = X
    ab = (lam @ a >= 0) & (lam @ b >= 0); ba = (lam @ b >= 0) & (lam @ a >= 0)
    order_ok = bool(np.array_equal(ab, ba))
    print(f"  (ii) порядок a·b == b·a (пересечение коммутирует): {order_ok}")
    # (iii) no-signaling оба порядка при p=2
    ns_ab = steer_scan(lam, 2.0)     # Боб-маргинал по настройке Алисы
    print(f"  (iii) no-signaling p=2 (Δ Боб по Алисе): {ns_ab:.4f} (<2σ={2*SIG:.4f}: {ns_ab<2*SIG})")
    # (iv) Δ-скан на ПОСЛЕ-измерительном ансамбле (после ŷ-преизмерения, s=+):
    # {λ_y>0} x-симметричен (a=x̂ тривиально) И ẑ-разложимо (не пусто).
    Y = np.array([0.0, 1.0, 0.0])
    post = lam[lam @ Y >= 0]
    zs = zero_set(post); Dpost = D_moment(post)
    # аналитика: λ_y>0 не меняет λ_z-маргинал ⇒ та же плотность 1+0.5c
    zs_an = analytic_zero_set(lambda c: 1 + 0.5*c)
    print(f"  (iv) послеизм. ансамбль (ŷ+): D={Dpost:.4f}, нуль-множество Δ числ={zs} аналит={zs_an}")
    scan = {f"{p}": steer_scan(post, p) for p in P_GRID}
    ok = (rep > 0.999) and order_ok and (ns_ab < 2*SIG) and (zs_an == [2.0])
    print(f"  T1 ВЕРДИКТ: {'ПРОХОД (репит+порядок+ns+нуль={2})' if ok else 'СТОП — разбор'}")
    return dict(rep=rep, order_ok=order_ok, ns_p2=ns_ab, D_post=None if np.isnan(Dpost) else Dpost,
                zero_set=zs, scan=scan, pass_=bool(ok))


def T2_deform():
    print("\n" + "="*72); print("T2 — анти-циркулярность (деформации меры)"); print("="*72)
    measures = {
        "mu_k0.5": lambda c: (1 + 0.5*c)**0.5,
        "mu_k2":   lambda c: (1 + 0.5*c)**2,
        "mu_k3":   lambda c: (1 + 0.5*c)**3,
        "mu_cub_a": lambda c: 1 + 0.5*c + 0.3*c**3,
        "mu_cub_b": lambda c: 1 + 0.3*c + 0.5*c**3,
    }
    out = {}
    allrobust = True
    print(f"  {'мера':10} {'D':>8} {'нуль-множ числ(2σ)':>20} {'аналит':>10} {'max|ан−чис|':>12}")
    for name, dens in measures.items():
        lam = sample_density(dens); D = D_moment(lam)
        zs = zero_set(lam); zs_an = analytic_zero_set(dens)
        mism = max(abs(analytic_Delta(dens, p) - steer_scan(lam, p)) for p in P_GRID)
        scan = {f"{p}": steer_scan(lam, p) for p in P_GRID}
        # ГЕЙТ per addendum2 (=prereg addendum1/A4): D(χ)>0 ∧ аналит. нуль-множ.={2}
        # (арбитр физики, в p=2 алгебраический ноль) ∧ |аналит−числ|<2σ.
        # Числ. нуль-множ. шире {2} (напр. 1.75 у mu_k0.5) = статистика, снимается
        # согласием <2σ; tol=1e-9-гейт на квадратуре (ложный СТОП 19:39) снят.
        robust = (D is not None and not np.isnan(D) and D > 5*SIG and zs_an == [2.0] and mism < 2*SIG)
        allrobust = allrobust and robust
        out[name] = dict(D=None if (D is None or np.isnan(D)) else float(D), zero_set_num=zs,
                         zero_set_analytic=zs_an, max_mismatch=float(mism), scan=scan, robust=bool(robust))
        print(f"  {name:10} {D:>8.4f} {str(zs):>20} {str(zs_an):>10} {mism:>12.5f}  {'✓' if robust else '⚠'}")
        if zs_an != [2.0] and D and D > 5*SIG:
            print(f"    ⚠⚠ СТОП: аналит. нуль-множество ≠ {{2}} при D>0 на {name} — сужение клейма!")
        elif zs != [2.0]:
            print(f"    (числ. нуль-множ {zs} шире {{2}} — статистика; аналитика {zs_an}={{2}} снимает)")
    print(f"  T2 ВЕРДИКТ: замыкание {'РОБАСТНО (аналит. нуль={2} на всех деформациях с D>0)' if allrobust else 'СУЖЕНО — разбор'}")
    return dict(measures=out, robust=bool(allrobust))


def T3_F1():
    print("\n" + "="*72); print("T3 — F1 (взвешенная хорда), структурная оценка"); print("="*72)
    # F1: P(s,t|a,b) ∝ |s√w_a a − t√w_b b|^p — ПРЯМОЙ joint-закон (не стиринг-коллапс).
    def setting(deg): t = np.radians(deg); return np.array([np.sin(t), 0, np.cos(t)])
    def chsh_ns(p, wa, wb):
        def E(al, be):
            a, b = setting(al), setting(be); br = [(1,1),(1,-1),(-1,1),(-1,-1)]
            W = np.array([np.linalg.norm(s*np.sqrt(wa)*a - t*np.sqrt(wb)*b)**p for s,t in br]); W/=W.sum()
            return sum(W[i]*br[i][0]*br[i][1] for i in range(4))
        # no-signaling: P(t|b) при разных a
        def Pt(al, be):
            a, b = setting(al), setting(be); br = [(1,1),(1,-1),(-1,1),(-1,-1)]
            W = np.array([np.linalg.norm(s*np.sqrt(wa)*a - t*np.sqrt(wb)*b)**p for s,t in br]); W/=W.sum()
            return W[0]+W[2]  # t=+1 (branches (1,1),(-1,1))
        ns = abs(Pt(0,60) - Pt(120,60))
        S = abs(E(0,45)-E(0,135)+E(90,45)+E(90,135))
        return ns, S
    print(f"  {'p':>4} {'wa,wb=1,1':>18} {'wa,wb=1.4,0.7 (χ)':>22}")
    out = {}
    for p in [1.0, 2.0, 3.0]:
        ns1, S1 = chsh_ns(p, 1.0, 1.0); nsx, Sx = chsh_ns(p, 1.4, 0.7)
        out[f"p{p}"] = dict(ns_sym=ns1, S_sym=S1, ns_asym=nsx, S_asym=Sx)
        print(f"  {p:>4} ns={ns1:.4f} S={S1:.3f}   ns={nsx:.4f} S={Sx:.3f}")
    print("  ОЦЕНКА: F1 — ПРЯМОЙ корреляционный закон (p — свободный параметр семьи),")
    print("  НЕ стиринг-коллапс модель ⇒ B5 (внутр. генерация стиринга) НЕ применима к F1")
    print("  тем же способом, что к F2. F1 не порождает партиально-запутанные ансамбли")
    print("  из скрытой меры — она ИХ ПОСТУЛИРУЕТ. ⇒ F1 не закрывает импорт как F2;")
    print("  фиксируется ГРАНИЦЕЙ конструкции (per addendum: провал F1 не рушит теорему).")
    return dict(chsh_ns=out, closes_import=False,
                note="F1 direct-law, B5 N/A; boundary result, F2 carries the theorem")


def main():
    t1 = T1_chain(); t2 = T2_deform(); t3 = T3_F1()
    json.dump(dict(T1=t1, T2=t2, T3=t3, sigma_bin=SIG, p_grid=P_GRID),
              open(os.path.join(RES, "C3B_hardening.json"), "w"), indent=2)
    print(f"\n  → {RES}/C3B_hardening.json")
    print(f"\n=== СТАТУС КЛЕЙМА: T1 {'✓' if t1['pass_'] else '✗'} T2 {'✓' if t2['robust'] else '✗'} "
          f"⇒ {'internal derivation of Born (steering class)' if (t1['pass_'] and t2['robust']) else 'closure (single measurements)'}")


if __name__ == "__main__":
    main()
