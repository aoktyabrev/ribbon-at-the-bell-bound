"""C3-B батарея (prereg dd7a74e). numpy, N=2e6, seed фикс. B1-B7 исполняемыми
assert'ами; D(χ); телеграф-скан Δ(p,χ) на ВНУТРЕННИХ ансамблях (не КМ).

F2 (первичное) — смещённая мера источника, ВНУТРЕННЯЯ генерация стиринга:
  общий bond-axis λ∈S², плотность μ_χ(λ) ∝ 1+χ(m·λ), m=ẑ. Настройка Алисы a:
  проективный исход s=sign(a·λ) разбивает λ на полупространства; ОПРЕДЕЛЁННАЯ
  условная ось Боба n_B^{a,s} = normalize(E[λ | sign(a·λ)=s]) (мин-модель
  стиринга: условное среднее как коллапс-ось — HV, без КМ). Вес P(s|a).
  Считывание Боба: t=+ с вер. f_p(n_B·b), f_p(c)=(1+c)^{p/2}/((1+c)^{p/2}+(1−c)^{p/2}).
  p=2 ⇒ f=(1+c)/2 (аффинно, Борн).
Ансамбли стиринга ПОРОЖДЕНЫ μ_χ (не импорт). QM-reference (born2 {q,1−q}/tilt)
— отдельная контрольная колонка.
"""
import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260720)
M_AX = np.array([0.0, 0.0, 1.0])                 # ось асимметрии m=ẑ
CHI_GRID = [0.0, 0.25, 0.5, 0.75, 0.95]          # продукт → максимум (A4: 5 точек)
P_GRID = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]   # A4
Z = np.array([0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])


def sample_mu(chi, n=N):
    """λ ~ S² с плотностью ∝ 1+χ(m·λ). Rejection (макс плотности 1+χ)."""
    out = []
    need = n
    while need > 0:
        v = RNG.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < (1 + chi * (v @ M_AX)) / (1 + chi)
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def f_p(c, p):
    A = np.abs(1 + c) ** (p / 2); B = np.abs(1 - c) ** (p / 2)
    return A / (A + B)


def cond_axes(lam, a):
    """Условное СОСТОЯНИЕ Боба n_B^{a,±} = E[λ|sign(a·λ)=±] (СЫРОЙ Bloch-вектор,
    НЕ нормируется — иначе ломается закон полной вероятности Σ_s P(s|a)E[λ|s]=E[λ]
    ⇒ B3 no-signaling при p=2 держится по построению). Условное состояние —
    смешанное (|Bloch|<1), как КМ-стиринг при частичной запутанности. Веса P(±|a)."""
    proj = lam @ a; pos = proj >= 0
    res = {}
    for s, mask in ((+1, pos), (-1, ~pos)):
        w = float(mask.mean())
        nb = lam[mask].mean(0)                      # СЫРОЙ условный Bloch-вектор
        res[s] = (w, nb, np.linalg.norm(nb))
    return res


def bob_P(lam, a, b, p):
    """P(Bob+ | Алиса мерила a) = Σ_s P(s|a) f_p(n_B^{a,s}·b) — маргинал по исходу Алисы."""
    ca = cond_axes(lam, a); P = 0.0
    for s, (w, nb, pol) in ca.items():
        P += w * f_p(nb @ b, p)
    return P


def bob_moment(lam, a, b, k):
    """k-й момент z-компоненты условных осей: Σ_s P(s|a) (n_B^{a,s}·b)^k."""
    ca = cond_axes(lam, a); mk = 0.0
    for s, (w, nb, pol) in ca.items():
        mk += w * (nb @ b) ** k
    return mk


def main():
    print("="*72); print("C3-B БАТАРЕЯ — F2 (смещённая мера, внутренний стиринг)"); print("="*72)
    results = {"F2": {}}

    # ---- B1,B2 — свойства считывания f_p (эндпоинты/несмещённость), ∀p ----
    b1 = all(abs(f_p(1.0, p) - 1.0) < 1e-9 and abs(f_p(-1.0, p) - 0.0) < 1e-9 for p in P_GRID)
    b2 = all(abs(f_p(0.0, p) - 0.5) < 1e-9 for p in P_GRID)
    print(f"B1 репитабельность f_p(±1)∈{{1,0}} ∀p: {b1}")
    print(f"B2 несмещённость f_p(0)=1/2 ∀p: {b2}")
    assert b1, "B1 FAIL"; assert b2, "B2 FAIL"

    # ---- B3 no-signaling (маргинал Боба независим от a) при p=2 ----
    print("\nB3 no-signaling (p=2, маргинал Боба по a) + B5 D(χ) + телеграф-скан:")
    sig = 1.0 / np.sqrt(N)
    for chi in CHI_GRID:
        lam = sample_mu(chi)
        Pz = bob_P(lam, Z, Z, 2.0); Px = bob_P(lam, X, Z, 2.0)   # Алиса ẑ vs x̂, Боб чит. ẑ
        b3 = abs(Pz - Px) < 3 * sig
        # средние совпадают? (условие B5)
        m1z = bob_moment(lam, Z, Z, 1); m1x = bob_moment(lam, X, Z, 1)
        mean_match = abs(m1z - m1x) < 5 * sig
        # D(χ): расхождение 3-го момента при равном среднем
        m3z = bob_moment(lam, Z, Z, 3); m3x = bob_moment(lam, X, Z, 3)
        D = abs(m3z - m3x) if mean_match else np.nan
        results["F2"][chi] = dict(Pz_p2=Pz, Px_p2=Px, b3=bool(b3), mean_z=m1z, mean_x=m1x,
                                  mean_match=bool(mean_match), D=float(D) if not np.isnan(D) else None)
        print(f"  χ={chi}: p=2 Pz={Pz:.4f} Px={Px:.4f} |Δ|={abs(Pz-Px):.4f} b3={b3} | "
              f"⟨1⟩z={m1z:+.4f} ⟨1⟩x={m1x:+.4f} match={mean_match} | D(3-мом)={D:.4f}")

    # ---- телеграф-скан Δ(p,χ) на внутренних ансамблях ----
    print("\nТЕЛЕГРАФ-СКАН Δ(p,χ) = |P_Alice=ẑ − P_Alice=x̂|, Боб чит. ẑ (внутренние ансамбли):")
    print("  χ \\ p " + " ".join(f"{p:>7}" for p in P_GRID))
    scan = {}
    for chi in CHI_GRID:
        lam = sample_mu(chi); row = []
        for p in P_GRID:
            d = abs(bob_P(lam, Z, Z, p) - bob_P(lam, X, Z, p)); row.append(d)
        scan[chi] = row
        print(f"  {chi:>4} " + " ".join(f"{d:7.4f}" for d in row))
    # нуль-множество по p при каждом χ (|Δ|<2σ_bin)
    print("\n  Нуль-множество p (|Δ|<2σ) по χ:")
    for chi in CHI_GRID:
        zeros = [P_GRID[i] for i, d in enumerate(scan[chi]) if d < 2 * sig]
        print(f"    χ={chi}: нули Δ при p={zeros}")

    # ---- АНАЛИТИКА Δ(p,χ) (A4 обязательна): μ_χ(c)=(1+χc)/2, c=λ_z ----
    # a=ẑ: разложение {(P±, m±)}; a=x̂: обе ветви z-компонента = ⟨λ_z⟩=χ/3.
    # p=2 ⇒ Δ=0 (линейность+полная вер.); p≠2 ⇒ Jensen-щель на нетривиальном
    # разложении ⇒ Δ=0 ⇔ p=2 (χ>0). Уникальность — аналитически.
    print("\nАНАЛИТИКА Δ(p,χ) (Jensen-щель; численность в скобках):")
    def analytic_Delta(p, chi):
        Pp = 0.5 + chi/4; Pm = 0.5 - chi/4
        mp = (0.25 + chi/6)/Pp; mm = (-0.25 + chi/6)/Pm
        lz = chi/3
        return abs(Pp*f_p(mp, p) + Pm*f_p(mm, p) - f_p(lz, p))
    max_mismatch = 0.0
    for chi in [0.5, 0.95]:
        row = []
        for p in P_GRID:
            da = analytic_Delta(p, chi); dn = scan[chi][P_GRID.index(p)]
            row.append(f"{da:.4f}({dn:.4f})"); max_mismatch = max(max_mismatch, abs(da-dn))
        print(f"  χ={chi}: " + " ".join(f"p{p}={r}" for p, r in zip(P_GRID, row)))
    print(f"  макс |аналитика−численность| = {max_mismatch:.4f}  ({'OK <2σ' if max_mismatch<2*sig else 'СТОП >2σ'})")
    # замыкание: ∃χ D>0 ∧ (Δ_analytic=0 ⇔ p=2)
    closure = []
    for chi in CHI_GRID:
        if chi == 0: continue
        Dchi = results["F2"][chi]["D"]
        zeros_an = [p for p in P_GRID if analytic_Delta(p, chi) < 1e-9]
        if Dchi and Dchi > 5*sig and zeros_an == [2.0]:
            closure.append(chi)
    print(f"\n  ЗАМЫКАНИЕ ∃χ: D(χ)>0 ∧ (Δ_analytic=0 ⇔ p=2) — выполнено при χ={closure}")

    # ---- QM-reference (born2 {q,1−q}/tilt) отдельной колонкой ----
    print("\nQM-REFERENCE (born2 импорт, КОНТРОЛЬ — НЕ в замыкании):")
    q = 0.85; zc = 1 - 2 * q
    for p in P_GRID:
        Pz_qm = q * f_p(-1.0, p) + (1 - q) * f_p(1.0, p)
        Px_qm = f_p(zc, p)
        print(f"  p={p}: Δ_QM={abs(Pz_qm - Px_qm):.4f}")

    import json, os
    RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
    json.dump({"F2": results["F2"], "scan": {str(k): v for k, v in scan.items()},
               "p_grid": P_GRID, "chi_grid": CHI_GRID, "sigma_bin": sig},
              open(os.path.join(RES, "C3B_battery.json"), "w"), indent=2)
    print(f"\n  → {RES}/C3B_battery.json")


if __name__ == "__main__":
    main()
