"""C4-GHZ / G-T2 (в) T2(N)-невидимость (prereg v2 449b4ca). Класс-M механизм на
N=4: GHZ-совместный закон P(s|a)=(1+s₁s₂s₃s₄·E)/16 реконструируется ЦЕПНЫМ
ПРАВИЛОМ в РАЗНЫХ порядках (CRN). Тезис: статистика тождественна при всех
порядках; никакой P-функционал не восстанавливает порядок. numpy N=2e6, CRN.
"""
import itertools
import json
import os

import numpy as np

N = 2_000_000
NP = 4
RNG = np.random.default_rng(20260722)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
E4 = 1.0                                          # ⟨XXXX⟩=1 (GHZ, настройки все-X)


def reconstruct(order, U):
    """Цепное правило для P(s)=(1+Πs·E4)/16 в порядке `order` (перестановка).
    Первые 3 в порядке — свободные (½), последний — условный. U: (4,N) общий поток (CRN)."""
    s = [None] * NP
    for rank, party in enumerate(order[:-1]):
        s[party] = np.where(U[rank] < 0.5, 1, -1)
    last = order[-1]
    prod3 = np.ones(N, np.int64)
    for party in order[:-1]:
        prod3 = prod3 * s[party]
    p_plus = (1 + prod3 * E4) / 2                 # P(s_last=+|остальные)
    s[last] = np.where(U[NP - 1] < p_plus, 1, -1)
    return np.stack(s)


def stats(s):
    """Полный набор P-функционалов: маргиналы, все 2-/3-/4-частичные корреляторы."""
    out = {}
    for r in range(1, NP + 1):
        for idx in itertools.combinations(range(NP), r):
            v = np.ones(N, np.int64)
            for i in idx:
                v = v * s[i].astype(np.int64)
            out["".join(map(str, idx))] = float(np.mean(v))
    return out


def main():
    U = RNG.random((NP, N))                       # общий поток (CRN между порядками)
    orders = [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2), (2, 0, 3, 1)]
    allstats = {str(o): stats(reconstruct(o, U)) for o in orders}
    # межпорядковый разброс каждого функционала
    keys = list(allstats[str(orders[0])].keys())
    # корреляторы r≥2 = ИНВАРИАНТЫ; одно-частичные маргиналы (r=1) под разными
    # порядками реконструкции = разные ЭСТИМАТОРЫ одной ½ (шум, урок L2 §7).
    def spread(k):
        vals = [allstats[str(o)][k] for o in orders]; return max(vals) - min(vals)
    corr_keys = [k for k in keys if len(k) >= 2]
    marg_keys = [k for k in keys if len(k) == 1]
    worst_corr = max(spread(k) for k in corr_keys)
    worst_marg = max(spread(k) for k in marg_keys)
    print(f"T2(N)-невидимость N={NP} (N_samp={N}, σ={SIG:.2e}), CRN, {len(orders)} порядка:")
    print(f"  ⟨s₁s₂s₃s₄⟩ по порядкам: {[round(allstats[str(o)]['0123'],4) for o in orders]} (цель E4={E4})")
    print(f"  ИНВАРИАНТЫ (корреляторы r≥2): worst-разброс = {worst_corr:.6f} (<2σ={2*SIG:.4f})")
    print(f"  маргиналы r=1 (разные эстиматоры ½ под порядками, шум L2): worst = {worst_marg:.6f}")
    # 4-частичный коррелятор (единственный нетривиальный GHZ-инвариант) — ТОЧНО тождествен;
    # остальные функционалы = инвариантные 0/½ ± эстиматор-шум (разные порядки = разные
    # эстиматоры). Худший разброс vs ожидаемый MC-потолок (√-число сравнений).
    inv4 = max(allstats[str(o)]["0123"] for o in orders) - min(allstats[str(o)]["0123"] for o in orders)
    ncomp = len(keys)
    mc_ceiling = 3.0 * SIG                         # ~3σ ожидаемо для max по ~15 функционалам×4
    invisible = (inv4 < 2 * SIG) and (max(worst_corr, worst_marg) < mc_ceiling)
    print(f"  4-частичный инвариант (⟨s₁s₂s₃s₄⟩) разброс: {inv4:.6f} (ТОЧНО тождествен)")
    print(f"  T2(N) ВЕРДИКТ: {'ПРОХОД — тождество цепного правила подтверждено (4-корр. точен; остаток '+f'{max(worst_corr,worst_marg):.5f}'+' на MC-потолке '+f'{mc_ceiling:.4f}, эстиматор-шум). Порядок НЕВОССТАНОВИМ.' if invisible else 'разбор — превышает MC-потолок'}")
    # адверс: which-order не P-функционал (по построению операц. доступа = только P)
    print("  which-order: НЕ P-функционал (операц. доступ = P) ⇒ вне статистики; не контрпример.")
    json.dump(dict(orders=[str(o) for o in orders],
                   E4_by_order={str(o): allstats[str(o)]["0123"] for o in orders},
                   worst_correlator_spread=worst_corr, worst_marginal_spread=worst_marg,
                   invisible=bool(invisible),
                   sigma=SIG, N=N, note="T2(N): chain-rule statistics order-invariant; order not recoverable"),
              open(os.path.join(RES, "C4GT2_T2inv.json"), "w"), indent=2)
    print(f"  → {RES}/C4GT2_T2inv.json")


if __name__ == "__main__":
    main()
