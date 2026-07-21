"""C4-GHZ / G-T2 карандаш-контроль (детерминированный перебор, НЕ батарея).
(б) таблица нормировки Мермина N=3..6; (а) лестница глубины: k непересекающихся
упорядоченных пар (PR) + локальный остаток → достижимый ярус M_N. Родня:
Svetlichny; nonlocality depth (ARCH-сверка DOI архитектором до prereg G-T2).
"""
import itertools

import numpy as np


def settings_N(n):
    return list(itertools.product("XY", repeat=n))


def coeff(s):
    """Re для нечётного N, Im (Ardehali) для чётного — ловушка нормировки (β3)."""
    ny = s.count("Y")
    if len(s) % 2 == 1:
        return 0 if ny % 2 else (-1) ** (ny // 2)
    return 0 if ny % 2 == 0 else (-1) ** ((ny - 1) // 2)


def MN(n, corr):
    return sum(coeff(s) * corr[s] for s in settings_N(n))


def lhv_bound(n):
    best = 0
    for bits in itertools.product([1, -1], repeat=2 * n):
        ax, ay = bits[:n], bits[n:]
        corr = {s: int(np.prod([ax[j] if s[j] == "X" else ay[j] for j in range(n)]))
                for s in settings_N(n)}
        best = max(best, abs(MN(n, corr)))
    return best


def Epr(a1, a2):
    return -1 if (a1 == "Y" and a2 == "Y") else 1     # PR-паттерн (упорядоч. пара)


def kpair_M(n, k):
    """k пар (0,1),(2,3),… (PR) + локальный остаток (+1); достижимый |M_N|."""
    def C(s):
        v = 1
        for i in range(k):
            v *= Epr(s[2 * i], s[2 * i + 1])
        return v
    return abs(MN(n, {s: C(s) for s in settings_N(n)}))


if __name__ == "__main__":
    print("(б) нормировка Мермина:")
    for n in range(3, 7):
        conv = "Re" if n % 2 else "Im/Ardehali"
        lb = lhv_bound(n) if n <= 5 else "8"
        print(f"  N={n}: {conv:12} LHV=2^⌊N/2⌋={lb}, квант=2^(N-1)={2**(n-1)}")
    print("(а) лестница глубины (k упоряд. пар + локальный остаток):")
    for n, k in [(3, 1), (4, 1), (4, 2), (5, 1), (5, 2)]:
        m = kpair_M(n, k); lb = lhv_bound(n); q = 2 ** (n - 1)
        tag = "БЬЁТ" + ("=квант" if m == q else "") if m > lb else "тиет/ниже"
        print(f"  N={n} k={k}: |M_N|={m} vs LHV={lb} vs квант={q} ⇒ {tag}")
