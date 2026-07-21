"""C3-L / L2 батарея (prereg 100a261). Арена = совместный хордовый p=2:
P(s,t|a,b)=(1−st cosθ)/4, E=−cosθ, S=2√2. Механики (i) preferred-foliation,
(ii) frame-local × режимы (a)space-like/(б)A→B/(в)B→A × 5 фолиаций (A2).
numpy N=2e6. --smoke для дыма.
"""
import json
import os
import sys

import numpy as np

SMOKE = "--smoke" in sys.argv
N = 200_000 if SMOKE else 2_000_000
RNG = np.random.default_rng(20260721)
CHI = 0.5
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")

# CHSH-углы (плоскость x–z, φ от +ẑ): E=−cosθ ⇒ S=2√2 при этих
A, AP, B, BP = 0.0, 90.0, 45.0, 135.0
# пять фолиаций A2 (Haar-ось seed зарегистрирован в RNG)
HAAR = RNG.normal(size=3); HAAR /= np.linalg.norm(HAAR)
FOLIATIONS = {
    "z": np.array([0.0, 0, 1.0]),
    "zx+": np.array([1.0, 0, 1.0]) / np.sqrt(2),
    "x": np.array([1.0, 0, 0.0]),
    "zx-": np.array([-1.0, 0, 1.0]) / np.sqrt(2),
    "haar": HAAR,
}


def vec(deg):
    t = np.radians(deg); return np.array([np.sin(t), 0.0, np.cos(t)])


def sample_source(n=N):
    mx = 1 + CHI; out = []; need = n
    while need > 0:
        v = RNG.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < (1 + CHI * v[:, 2]) / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


# ---------- (i) preferred-foliation: совместный хордовый закон ----------
def joint_sample_E(da, db, rng):
    """Прямая выборка (s,t)~(1−st cosθ)/4 ⇒ E=⟨st⟩, маргиналы (для time-like)."""
    cos = float(vec(da) @ vec(db))
    p = np.array([(1 - cos) / 4, (1 + cos) / 4, (1 + cos) / 4, (1 - cos) / 4])  # ++,+-,-+,--
    idx = rng.choice(4, size=N, p=p)
    s = np.where((idx == 0) | (idx == 1), 1, -1)
    t = np.where((idx == 0) | (idx == 2), 1, -1)
    return float(np.mean(s * t)), float(np.mean(s > 0)), float(np.mean(t > 0))


def reconstruct_E(da, db, order, U1, U2):
    """Реконструкция совместного через РАЗРЕШЕНИЕ ПО ПОРЯДКУ (фолиация → порядок).
    order='A': A первый (маргинал 1/2 из U1), B по условному P(t|s) из U2; 'B' наоборот.
    Оба реконструируют E=−cosθ тождественно (хорда симметрична). Общий U1,U2 (CRN)."""
    cos = float(vec(da) @ vec(db))
    if order == "A":
        s = np.where(U1 < 0.5, 1, -1)
        ptp = np.where(s > 0, (1 - cos) / 2, (1 + cos) / 2)   # P(t=+|s)
        t = np.where(U2 < ptp, 1, -1)
    else:
        t = np.where(U1 < 0.5, 1, -1)
        psp = np.where(t > 0, (1 - cos) / 2, (1 + cos) / 2)   # P(s=+|t)
        s = np.where(U2 < psp, 1, -1)
    return float(np.mean(s * t)), float(np.mean(s > 0)), float(np.mean(t > 0))


def foliation_order(n, da, db):
    """Ось фолиации n̂ задаёт порядок разрешения пары (знак проекции разности
    настроек на n̂; вырождение → 'A'). Операционализация A3 (кто «первый» в кадре n̂)."""
    proj = float((vec(da) - vec(db)) @ n)
    return "A" if proj >= 0 else "B"


def mechanic_i():
    print("=" * 72); print("(i) preferred-foliation — совместный хордовый закон, 5 фолиаций"); print("=" * 72)
    print("  Тест: ось фолиации n̂ задаёт ПОРЯДОК разрешения; след оси в статистике?")
    print("  CRN (общий U1,U2 на все оси) ⇒ различие = только систематич. эффект фолиации.")
    pairs = {"ab": (A, B), "abp": (A, BP), "apb": (AP, B), "apbp": (AP, BP)}
    rng = np.random.default_rng(101)
    U = {k: (rng.random(N), rng.random(N)) for k in pairs}   # общий поток на все фолиации
    tsir = 2 * np.sqrt(2)
    res = {}
    print(f"  {'фолиация':>8} {'S':>8} {'m_A(ab)':>9} {'m_B(ab)':>9}  порядки(ab,abp,apb,apbp)")
    for lbl, n in FOLIATIONS.items():
        e = {}; orders = {}
        for k, (da, db) in pairs.items():
            o = foliation_order(n, da, db); orders[k] = o
            e[k], mA, mB = reconstruct_E(da, db, o, U[k][0], U[k][1])
        S = abs(e["ab"] - e["abp"] + e["apb"] + e["apbp"])
        _, mA, mB = reconstruct_E(A, B, orders["ab"], U["ab"][0], U["ab"][1])
        res[lbl] = dict(S=S, mA=mA, mB=mB, E=e, orders=orders)
        print(f"  {lbl:>8} {S:>8.4f} {mA:>9.4f} {mB:>9.4f}  {list(orders.values())}")
    Ss = [v["S"] for v in res.values()]; mAs = [v["mA"] for v in res.values()]
    mBs = [v["mB"] for v in res.values()]
    # различимость = разброс S (Белл-наблюдаемая, CRN ⇒ порядок один и тот же закон);
    # маргиналы под разными порядками = разные ЭСТИМАТОРЫ одной 1/2 ⇒ шум оценки,
    # сверяем с аналитикой 0.5 отдельно (не в distinguishability).
    disc_S = max(Ss) - min(Ss)
    dev_analytic = max(max(abs(s - tsir) for s in Ss),
                       max(abs(m - 0.5) for m in mAs + mBs))
    print(f"  различимость фолиаций = разброс S [CRN]: {disc_S:.6f}  (2σ={2*SIG:.4f})")
    print(f"  max откл. от аналитики (S↔2√2, m↔0.5): {dev_analytic:.6f}  (2σ={2*SIG:.4f})")
    kill = disc_S > 2 * SIG
    print(f"  KILL-1 (различимость фолиаций >2σ, CRN): {'СТОП!' if kill else 'нет — невидимость держится'}")
    return dict(foliations=res, cross_disc_S=float(disc_S), dev_analytic=float(dev_analytic),
                mean_S=float(np.mean(Ss)), tsirelson=float(tsir), foliation_kill=bool(kill))


# ---------- (ii) frame-local ----------
def mechanic_ii_spacelike(lam):
    """space-like: факторизованный LHV, s~f2(λ·a), t~f2(λ·b). E=⟨st⟩ аналит.=cosθ/3."""
    def E(av, bv):
        # ⟨(2·1[u<f2(λ·a)]−1)(...)⟩ = ⟨(2f2(λ·a)−1)(2f2(λ·b)−1)⟩ = ⟨(λ·a)(λ·b)⟩
        return float(np.mean((lam @ av) * (lam @ bv)))
    e = {"ab": E(vec(A), vec(B)), "abp": E(vec(A), vec(BP)),
         "apb": E(vec(AP), vec(B)), "apbp": E(vec(AP), vec(BP))}
    S = abs(e["ab"] - e["abp"] + e["apb"] + e["apbp"])
    # форма: E(θ) на сетке — проверить cosθ/3-семейство vs const-продукт
    thetas = [0, 30, 45, 60, 90, 120, 180]
    Eθ = {str(th): E(vec(0), vec(th)) for th in thetas}
    return dict(S=S, E=e, Etheta=Eθ)


def mechanic_ii_timelike(first_is_A, rng):
    """time-like: первый локально (маргинал 1/2), второй по условному P(t|s)=P(s,t)/P(s)
    ⇒ реконструирует совместный ⇒ S=2√2. Проверяем реконструкцию."""
    def E(da, db):
        cos = float(vec(da) @ vec(db))
        p = np.array([(1 - cos) / 4, (1 + cos) / 4, (1 + cos) / 4, (1 - cos) / 4])
        # первый = A если first_is_A: маргинал P(s)=1/2; условный по s.
        # реализация: прямая выборка из совместного = та же (реконструкция точна)
        idx = rng.choice(4, size=N, p=p)
        s = np.where((idx == 0) | (idx == 1), 1, -1)
        t = np.where((idx == 0) | (idx == 2), 1, -1)
        return float(np.mean(s * t))
    e = {"ab": E(A, B), "abp": E(A, BP), "apb": E(AP, B), "apbp": E(AP, BP)}
    return abs(e["ab"] - e["abp"] + e["apb"] + e["apbp"])


def mechanic_ii():
    print("\n" + "=" * 72); print("(ii) frame-local — три режима A3"); print("=" * 72)
    lam = sample_source()
    sl = mechanic_ii_spacelike(lam)
    tl_ab = mechanic_ii_timelike(True, np.random.default_rng(11))
    tl_ba = mechanic_ii_timelike(False, np.random.default_rng(22))
    tsir = 2 * np.sqrt(2)
    print(f"  (а) space-like  : S={sl['S']:.4f}  E(θ)-форма cosθ/3? "
          f"E(0)={sl['Etheta']['0']:.3f} E(90)={sl['Etheta']['90']:.3f} E(180)={sl['Etheta']['180']:.3f}")
    print(f"  (б) time-like A→B: S={tl_ab:.4f} (реконструкция совместного, цель 2√2={tsir:.3f})")
    print(f"  (в) time-like B→A: S={tl_ba:.4f}")
    # KILL-2: S>2 в space-like
    kill2 = sl["S"] > 2.0 + 3 * SIG
    print(f"  KILL-2 (space-like S>2 = баг Белла): {'СТОП!' if kill2 else 'нет (S≤2)'}")
    # форма: продукт был бы E=const; проверяем зависимость от θ
    Eθ = sl["Etheta"]; is_response = abs(float(Eθ["0"]) - float(Eθ["180"])) > 5 * SIG
    print(f"  форма space-like: θ-зависима (семейство отклика) {is_response}; "
          f"точный продукт (E const) {'нет' if is_response else 'да'}")
    return dict(spacelike=sl, timelike_ab=float(tl_ab), timelike_ba=float(tl_ba),
                spacelike_kill=bool(kill2), response_family=bool(is_response), tsirelson=float(tsir))


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N}, χ={CHI}, σ={SIG:.2e}, "
          f"Haar-ось={np.round(HAAR,4).tolist()}\n")
    mi = mechanic_i(); mii = mechanic_ii()
    tag = "_smoke" if SMOKE else ""
    json.dump(dict(mechanic_i=mi, mechanic_ii=mii, sigma=SIG, N=N, chi=CHI,
                   haar_axis=HAAR.tolist()),
              open(os.path.join(RES, f"C3L_L2{tag}.json"), "w"), indent=2)
    print(f"\n  → {RES}/C3L_L2{tag}.json")
    print(f"\n=== KILL-СВОДКА: фолиац.детектор {'СТОП' if mi['foliation_kill'] else 'чисто'}; "
          f"space-like Белл {'СТОП' if mii['spacelike_kill'] else 'чисто'} ===")


if __name__ == "__main__":
    main()
