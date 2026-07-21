"""C4-GHZ / G-T β-батарея (prereg edc5e6d). GHZ-механизм цепным правилом:
закон P(s₁s₂s₃|m)=(1+s₁s₂s₃·E₁₂₃)/8 (импорт КМ-закона ЛЕГАЛЕН — тест ТЕОРЕМЫ
О КЛАССЕ, не внутренности). Порядки: 6 перестановок + пер-рановая монета (CRN).
Проверка: M₃=4, попарные=0, маргиналы ½, порядко-/монето-невидимость. numpy N=2e6.
"""
import itertools
import json
import os

import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260721)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")

# GHZ-корреляторы (X=0,Y=1): E₁₂₃ = +1 (XXX), −1 (две Y), 0 (одна/три Y).
def E123(m):
    ny = sum(m)
    if ny == 0:
        return 1.0            # XXX
    if ny == 2:
        return -1.0           # XYY,YXY,YYX
    return 0.0                # одна или три Y


def resolve(m, order, U):
    """Цепное правило для P(s₁s₂s₃|m)=(1+s₁s₂s₃E)/8 в порядке `order` (перестановка
    индексов). Маргиналы и попарные = ½/0 ⇒ первые два свободны, третий условный.
    U: (3,N) общий поток (CRN). Возврат s (3,N)∈{±1}."""
    E = E123(m); n = U.shape[1]
    s = np.empty((3, n), np.int8)
    i0, i1, i2 = order
    s[i0] = np.where(U[0] < 0.5, 1, -1)
    s[i1] = np.where(U[1] < 0.5, 1, -1)
    # P(s_i2=+ | s_i0,s_i1) = (1 + s_i0 s_i1 (+1) E)/2
    p_plus = (1 + s[i0].astype(np.int64) * s[i1].astype(np.int64) * E) / 2
    s[i2] = np.where(U[2] < p_plus, 1, -1)
    return s


def corr(s, idx):
    v = np.ones(N, np.int64)
    for i in idx:
        v = v * s[i].astype(np.int64)
    return float(np.mean(v))


ORDERS = list(itertools.permutations([0, 1, 2]))
MERMIN = [((0, 0, 0), +1), ((0, 1, 1), -1), ((1, 0, 1), -1), ((1, 1, 0), -1)]  # (m, знак в M₃)


def mermin_M3(order, U_by_m):
    M = 0.0
    for m, sign in MERMIN:
        s = resolve(m, order, U_by_m[m])
        M += sign * corr(s, (0, 1, 2))
    return M


def main():
    print(f"β-монета GHZ N={N} σ={SIG:.2e}")
    # общий поток на каждую настройку (CRN между порядками/монетой)
    U_by_m = {m: RNG.random((3, N)) for m, _ in MERMIN}
    # --- Мермин по каждому фикс. порядку (6 перестановок) ---
    Ms = {}
    for o in ORDERS:
        Ms[str(o)] = mermin_M3(o, U_by_m)
    print("  M₃ по 6 порядкам разрешения:")
    for o, v in Ms.items():
        print(f"    {o}: M₃={v:+.4f}")
    disc_order = max(Ms.values()) - min(Ms.values())
    # --- пер-рановая монета: на реплику случайный порядок (общая, приготовит.) ---
    coin = RNG.integers(0, 6, N)
    M_coin = 0.0; s_ab_coin = None
    for m, sign in MERMIN:
        s = np.empty((3, N), np.int8)
        for oi, o in enumerate(ORDERS):
            mask = coin == oi
            if mask.any():
                sub = resolve(m, o, U_by_m[m][:, mask])
                s[:, mask] = sub
        M_coin += sign * corr(s, (0, 1, 2))
        if m == (0, 0, 0):
            s_ab_coin = s
    # --- GHZ-подпись: попарные + маргиналы (настройка XXX) ---
    s_xxx = resolve((0, 0, 0), (0, 1, 2), U_by_m[(0, 0, 0)])
    pair = [corr(s_xxx, (i, j)) for i, j in [(0, 1), (0, 2), (1, 2)]]
    marg = [float(np.mean(s_xxx[i] > 0)) for i in range(3)]
    # --- w-невидимость: M₃(монета) vs M₃(фикс. порядки) ---
    invis = abs(M_coin - np.mean(list(Ms.values())))
    tsir = 4.0
    print(f"  ⟨M₃⟩ по порядкам={np.mean(list(Ms.values())):.4f}; монета M₃={M_coin:.4f}")
    print(f"  попарные корреляции (XXX): {[round(p,5) for p in pair]} (GHZ→0)")
    print(f"  маргиналы: {[round(mm,4) for mm in marg]} (½)")
    print(f"  разброс M₃ по порядкам [CRN]={disc_order:.6f}; монето-невидимость={invis:.6f} (2σ={2*SIG:.4f})")
    checks = dict(
        M3_is_4=bool(abs(np.mean(list(Ms.values())) - tsir) < 2 * SIG),
        pairwise_zero=bool(max(abs(p) for p in pair) < 2 * SIG),
        marginals_half=bool(max(abs(mm - 0.5) for mm in marg) < 2 * SIG),
        order_invisible=bool(disc_order < 2 * SIG),
        coin_invisible=bool(invis < 2 * SIG))
    allpass = all(checks.values())
    print(f"  ВЕРДИКТ β-GHZ: {'ПРОХОД ЦЕЛИКОМ (M₃=4, попарн=0, порядок+монета невидимы)' if allpass else 'ПРОВАЛ: '+str(checks)}")
    json.dump(dict(M3_by_order={k: v for k, v in Ms.items()}, M3_coin=M_coin,
                   pairwise=pair, marginals=marg, order_disc=disc_order, coin_invis=invis,
                   checks=checks, all_pass=bool(allpass), sigma=SIG, N=N,
                   note="import of QM GHZ-law LEGAL: testing class-theorem, not law-internality"),
              open(os.path.join(RES, "C4GT_beta.json"), "w"), indent=2)
    print(f"  → {RES}/C4GT_beta.json")


if __name__ == "__main__":
    main()
