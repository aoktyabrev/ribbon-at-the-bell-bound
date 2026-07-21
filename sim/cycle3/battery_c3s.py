"""C3-S батарея J1-J7 (prereg 3d516d4). Амплитудный шов: семейства S-F1
(взвеш. совместная хорда α=√(1+χ),β=√(1−χ)) и S-F2 (смещ. источник ×
смесь хорда/продукт). Проверка зарегистр. кривых S(χ), no-signaling,
D_joint, единственность. numpy N=2e6, CRN, аналитика рядом. --smoke.
"""
import json
import os
import sys

import numpy as np

SMOKE = "--smoke" in sys.argv
N = 200_000 if SMOKE else 2_000_000
RNG = np.random.default_rng(20260721)
SIG = 1.0 / np.sqrt(N)
CHI_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]          # A4: 5 точек 0..χ_max^dop=1
P_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, np.inf]      # J5 единственность
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
TSIR = 2 * np.sqrt(2)

# CHSH-углы (плоскость x–z): E=−cosθ ⇒ S=2√2
ANG = dict(a=0.0, ap=90.0, b=45.0, bp=135.0)


def vec(deg):
    t = np.radians(deg); return np.array([np.sin(t), 0.0, np.cos(t)])


# ============ S-F1: взвешенная совместная хорда (постулированный закон) ============
def sf1_E(da, db, chi, p=2.0):
    """E(θ|χ) из P∝|sαa−tβb|^p. p=2 аналит.=−√(1−χ²)cosθ; общий p — численно по 4 ветвям."""
    a, b = vec(da), vec(db); al, be = np.sqrt(1 + chi), np.sqrt(1 - chi)
    br = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    if np.isinf(p):
        w = np.array([np.linalg.norm(s * al * a - t * be * b) for s, t in br])
        W = (w == w.max()).astype(float)  # ступень: макс-хорда
    else:
        W = np.array([np.linalg.norm(s * al * a - t * be * b) ** p for s, t in br])
    W = W / W.sum()
    E = sum(W[i] * br[i][0] * br[i][1] for i in range(4))
    mA = W[0] + W[1]  # s=+
    mB = W[0] + W[2]  # t=+
    return E, mA, mB


def sf1_S(chi, p=2.0):
    E_ab = sf1_E(ANG["a"], ANG["b"], chi, p)[0]
    E_abp = sf1_E(ANG["a"], ANG["bp"], chi, p)[0]
    E_apb = sf1_E(ANG["ap"], ANG["b"], chi, p)[0]
    E_apbp = sf1_E(ANG["ap"], ANG["bp"], chi, p)[0]
    return abs(E_ab - E_abp + E_apb + E_apbp)


# ============ S-F2: смещённый источник × смесь хорда/продукт ============
def sample_source(chi, n=N):
    if chi == 0:
        v = RNG.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True); return v
    mx = 1 + chi; out = []; need = n
    while need > 0:
        v = RNG.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < (1 + chi * v[:, 2]) / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def sf2_E(lam, da, db):
    """E = −⟨q(λ)⟩·cosθ, q=(1+λ_z)/2 (смесь хорда/продукт; продукт даёт E=0)."""
    cos = float(vec(da) @ vec(db)); q = (1 + lam[:, 2]) / 2
    return float(np.mean(q) * (-cos)), 0.5, 0.5  # маргиналы ½ (обе компоненты симметричны)


def sf2_S(lam):
    E_ab, _, _ = sf2_E(lam, ANG["a"], ANG["b"])
    E_abp, _, _ = sf2_E(lam, ANG["a"], ANG["bp"])
    E_apb, _, _ = sf2_E(lam, ANG["ap"], ANG["b"])
    E_apbp, _, _ = sf2_E(lam, ANG["ap"], ANG["bp"])
    return abs(E_ab - E_abp + E_apb + E_apbp)


def sf2_Djoint(lam, rng):
    """D_joint (A3): пост-измерит. условная мера λ при (s,a) — ИСТИННАЯ.
    Alice s ~ P(s|a,λ). В смеси хорда/продукт P(s|a,λ)=½ ∀λ (обе компоненты
    симметричны) ⇒ выбор s НЕ зависит от λ ⇒ условная λ = источник ⇒ D_joint≈0.
    Настройки a=ẑ(0) и a=x̂(90) — равные средние ⟨λ_z⟩ по симметрии (как C3-B).
    3-й и 5-й моменты λ_z условного ансамбля {s=+}, расхождение по настройке."""
    def cond_moment(a_deg, k):
        av = vec(a_deg)
        # ИСТИННАЯ P(s=+|a,λ)=½ (маргинал закона) ⇒ s ⟂ λ:
        s_plus = rng.random(len(lam)) < 0.5
        sub = lam[s_plus]
        # среднее и k-й момент проекции на ось смещения ẑ
        return float(np.mean(sub[:, 2] ** k))
    a1, a2 = 0.0, 90.0
    dmean = abs(cond_moment(a1, 1) - cond_moment(a2, 1))
    d3 = abs(cond_moment(a1, 3) - cond_moment(a2, 3))
    d5 = abs(cond_moment(a1, 5) - cond_moment(a2, 5))
    return float(d3), float(d5), float(dmean)


# ============ J-батарея ============
def run_family_curves():
    print("=" * 72); print("J1/J2/J7 — зарегистрированные кривые S(χ), E(θ)"); print("=" * 72)
    print(f"  {'χ':>5} | {'S-F1 числ':>10} {'S-F1 крив':>10} {'|Δ|':>8} | "
          f"{'S-F2 числ':>10} {'S-F2 крив':>10} {'|Δ|':>8}")
    rows = {}; max_mis1 = 0; max_mis2 = 0
    for chi in CHI_GRID:
        s1 = sf1_S(chi); c1 = TSIR * np.sqrt(1 - chi ** 2)
        lam = sample_source(chi); s2 = sf2_S(lam); c2 = np.sqrt(2) * (1 + chi / 3)
        m1, m2 = abs(s1 - c1), abs(s2 - c2)
        max_mis1 = max(max_mis1, m1); max_mis2 = max(max_mis2, m2)
        rows[f"{chi}"] = dict(SF1_num=s1, SF1_curve=c1, SF2_num=s2, SF2_curve=c2)
        print(f"  {chi:>5} | {s1:>10.4f} {c1:>10.4f} {m1:>8.5f} | {s2:>10.4f} {c2:>10.4f} {m2:>8.5f}")
    print(f"  кривые: max|числ−аналит| SF1={max_mis1:.5f} SF2={max_mis2:.5f} (2σ={2*SIG:.4f})")
    print(f"  J2 (S=2√2 при χ_max): SF1 max_S={max(r['SF1_num'] for r in rows.values()):.4f} "
          f"(=2√2 при χ=0), SF2 max_S={max(r['SF2_num'] for r in rows.values()):.4f} (<2√2 всегда)")
    return dict(rows=rows, max_mismatch_SF1=float(max_mis1), max_mismatch_SF2=float(max_mis2),
                curve_ok=bool(max(max_mis1, max_mis2) < 2 * SIG))


def run_nosignaling():
    print("\n" + "=" * 72); print("J3 — no-signaling маргиналов ∀χ (главный риск: телеграф)"); print("=" * 72)
    worst = 0.0
    for chi in CHI_GRID:
        # S-F1: маргинал m_A при разных настройках Боба (должен = ½)
        mA_b = sf1_E(ANG["a"], ANG["b"], chi)[1]; mA_bp = sf1_E(ANG["a"], ANG["bp"], chi)[1]
        d1 = abs(mA_b - mA_bp)
        lam = sample_source(chi)
        mA2_b = sf2_E(lam, ANG["a"], ANG["b"])[1]; mA2_bp = sf2_E(lam, ANG["a"], ANG["bp"])[1]
        d2 = abs(mA2_b - mA2_bp)
        worst = max(worst, d1, d2)
        print(f"  χ={chi}: S-F1 Δm_A(по настройке Боба)={d1:.6f}  S-F2 Δm_A={d2:.6f}")
    kill = worst > 2 * SIG
    print(f"  J3 ВЕРДИКТ: max Δмаргинал={worst:.6f} (2σ={2*SIG:.4f}) "
          f"⇒ {'СТОП! ТЕЛЕГРАФ' if kill else 'no-signaling держится'}")
    return dict(worst_marginal=float(worst), telegraph_kill=bool(kill))


def run_Djoint():
    print("\n" + "=" * 72); print("J4 — внутренний стиринг D_joint(χ) + контроль 5-й момент"); print("=" * 72)
    out = {}
    print(f"  {'χ':>5} {'S-F2 D3':>10} {'S-F2 D5':>10} {'ср.совпад':>10}  S-F1: постулир., нет непрер. меры ⇒ N/A")
    for chi in CHI_GRID:
        lam = sample_source(chi); d3, d5, dmean = sf2_Djoint(lam, np.random.default_rng(int(chi * 100) + 3))
        out[f"{chi}"] = dict(SF2_D3=d3, SF2_D5=d5, mean_match=bool(dmean < 5 * SIG))
        print(f"  {chi:>5} {d3:>10.5f} {d5:>10.5f} {str(dmean<5*SIG):>10}")
    sf2_pos = any(v["SF2_D3"] > 5 * SIG for v in out.values())
    print(f"  S-F2 D_joint>0 при χ>0: {sf2_pos}. Причина (анзац №1): P(s|a,λ)=½ ∀λ ⇒")
    print(f"    измерение Алисы НЕ обновляет λ ⇒ стиринг НЕ порождён ⇒ D_joint≈0.")
    print(f"  S-F1 D_joint=N/A (постулир. закон). ⇒ ОБА семейства проваливают J4-генерацию.")
    return dict(SF2=out, SF2_Djoint_positive=bool(sf2_pos), SF1_Djoint="N/A_postulated")


def run_uniqueness():
    print("\n" + "=" * 72); print("J5 — единственность Δ(p,χ) в совместном слое (S-F1)"); print("=" * 72)
    # для S-F1 «телеграф» = зависимость маргинала Боба от настройки Алисы при p≠2
    def gap(chi, p):
        mB_a = sf1_E(ANG["a"], ANG["b"], chi, p)[2]
        mB_ap = sf1_E(ANG["ap"], ANG["b"], chi, p)[2]
        return abs(mB_a - mB_ap)
    chi = 0.5; out = {}
    print(f"  χ={chi}: Δ_маргинал Боба по настройке Алисы, скан p:")
    zero_set = []
    for p in P_GRID:
        g = gap(chi, p); key = "inf" if np.isinf(p) else f"{p}"
        out[key] = float(g)
        if g < 2 * SIG:
            zero_set.append(key)
        print(f"    p={key:>4}: Δ={g:.6f} {'(=0)' if g<2*SIG else ''}")
    print(f"  нуль-множество (Δ<2σ): {zero_set}")
    all_p = len(zero_set) == len(P_GRID)
    if all_p:
        print("  ФИНДИНГ: нуль-множество = ВСЕ p ⇒ постулированный хордовый закон")
        print("    имеет РАВНОМЕРНЫЕ маргиналы ∀p (Σ_t симметрия) ⇒ no-signaling ∀p ⇒")
        print("    p=2 НЕ выделяется в ПОСТУЛИРОВАННОМ слое (семья |·|^p — все NS-боксы,")
        print("    S(p) до PR). Отбор p=2 (лемма L3) живёт в СТИРИНГ-слое, не здесь.")
        print("    ⇒ SCOPE флагмана L3 §2.3: «joint layer» = СТИРИНГ-эндовед, не постулир.")
    return dict(chi=chi, scan=out, zero_set=zero_set, all_p_nosignaling=bool(all_p),
                scope_flag_L3="joint-layer cut requires steering, not postulated |·|^p")


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N}, σ={SIG:.2e}\n")
    curves = run_family_curves(); ns = run_nosignaling()
    dj = run_Djoint(); uniq = run_uniqueness()
    tag = "_smoke" if SMOKE else ""
    json.dump(dict(curves=curves, nosignaling=ns, Djoint=dj, uniqueness=uniq,
                   sigma=SIG, N=N, chi_grid=CHI_GRID,
                   registered=dict(SF1="2√2·√(1−χ²)", SF2="√2·(1+χ/3)")),
              open(os.path.join(RES, f"C3S{tag}.json"), "w"), indent=2)
    print(f"\n  → {RES}/C3S{tag}.json")
    # вердикт шва
    sf1_reaches = abs(TSIR - max(r["SF1_num"] for r in curves["rows"].values())) < 3 * SIG
    sf2_reaches = abs(TSIR - max(r["SF2_num"] for r in curves["rows"].values())) < 3 * SIG
    closed = sf1_reaches and dj["SF2_Djoint_positive"] and sf2_reaches
    print(f"\n=== ШОВ (в терминах prereg):")
    print(f"  S-F1: достигает 2√2={sf1_reaches} (ТОЛЬКО χ=0, симметрия), но D_joint=N/A")
    print(f"        (постулир. закон, нет непрер. меры) ⇒ провал J4 (внутр. генерация).")
    print(f"  S-F2 (попытка №1): D_joint>0={dj['SF2_Djoint_positive']} (анзац не стирит),")
    print(f"        достигает 2√2={sf2_reaches} (S≤1.886) ⇒ провал J2 И J4.")
    print(f"  ЗАМЫКАНИЕ ШВА = {'ДА' if closed else 'НЕТ'}. Улика невозможности (2 семейства,")
    print(f"  зарег. анзацы). Структурная причина = Белл (внутр. генерация=LHV=S≤2). "
          f"Клейм 'невозможность' — только A5 α/β/γ ===")


if __name__ == "__main__":
    main()
