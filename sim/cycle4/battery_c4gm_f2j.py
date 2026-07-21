"""C4-GHZ / GM-F2j J-батарея (prereg 3298e54). Чётность ⊂λ + упорядоченная пара
(s₂ зависит от a₁) + третья локально σ₃·h. Развилка (i)-(iv). Ожидание: (i) M₃=4
при GHZ-подписи (попарные 0, маргиналы ½) — трипартитный шов пересечён парой.
numpy N=2e6, CRN.
"""
import itertools
import json
import os

import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260721)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
SETTINGS = list(itertools.product("XY", repeat=3))
MERMIN = [(("X", "X", "X"), +1), (("X", "Y", "Y"), -1), (("Y", "X", "Y"), -1), (("Y", "Y", "X"), -1)]


def w(a1, a2):
    return 1 if (a1 == "X" and a2 == "X") else -1


def source():
    s1 = np.where(RNG.random(N) < 0.5, 1, -1)
    s2 = np.where(RNG.random(N) < 0.5, 1, -1)
    s3 = s1 * s2                        # чётность +1
    return s1, s2, s3


def outcomes(sig, a, order="12"):
    σ1, σ2, σ3 = sig
    if order == "12":
        s1 = σ1; s2 = σ2 * w(a[0], a[1])            # пара: s2 зависит от a1 (упоряд.)
    else:
        s2 = σ2; s1 = σ1 * w(a[0], a[1])            # обратный порядок пары (CRN)
    s3 = σ3 * 1                                      # h=+1
    return np.stack([s1, s2, s3])


def corr(s, idx):
    v = np.ones(N, np.int64)
    for i in idx:
        v = v * s[i].astype(np.int64)
    return float(np.mean(v))


def main():
    sig = source()
    M3 = sum(sgn * corr(outcomes(sig, a), (0, 1, 2)) for a, sgn in MERMIN)
    M3_21 = sum(sgn * corr(outcomes(sig, a, "21"), (0, 1, 2)) for a, sgn in MERMIN)
    order_disc = abs(M3 - M3_21)
    pair_max = 0.0; tri_min = 1.0; marg_dev = 0.0
    for a in SETTINGS:
        s = outcomes(sig, a)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            pair_max = max(pair_max, abs(corr(s, (i, j))))
        tri_min = min(tri_min, abs(corr(s, (0, 1, 2))))
        for i in range(3):
            marg_dev = max(marg_dev, abs(float(np.mean(s[i] > 0)) - 0.5))
    # NS: маргинал party2 от a1; двухчастичный (1,2) от a3
    def m2(a):
        return float(np.mean(outcomes(sig, a)[1] > 0))
    ns2 = max(abs(m2(("X", "X", "X")) - m2((a1, "X", "X"))) for a1 in "XY")
    def e12(a):
        return corr(outcomes(sig, a), (0, 1))
    ns12 = max(abs(e12(("X", "X", "X")) - e12(("X", "X", a3))) for a3 in "XY")
    ns_worst = max(ns2, ns12)

    print(f"GM-F2j J-батарея (N={N}, σ={SIG:.2e}) — чётность ⊂λ + упорядоченная пара:")
    print(f"  J-Мермин: M₃ = {M3:.4f} (бонд 2, алгебр. max 4); order-21 = {M3_21:.4f}")
    print(f"  J-GHZ: max|попарные|={pair_max:.6f} (<2σ={2*SIG:.4f}); min|тройной|={tri_min:.4f}")
    print(f"  маргиналы: max|·−½|={marg_dev:.6f} (полная GHZ-подпись, ½ на всех 8)")
    print(f"  J-NS worst = {ns_worst:.6f}; порядко-невидимость пары [CRN] = {order_disc:.6f}")
    sign = (pair_max < 2 * SIG) and (tri_min > 1 - 2 * SIG) and (marg_dev < 2 * SIG)
    exceeds = abs(M3) > 2 + 2 * SIG
    ns_ok = ns_worst < 2 * SIG
    if not ns_ok:
        fork = "(iv) NS-НАРУШЕНИЕ — СТОП-разбор"
    elif exceeds and sign:
        fork = "(i) M₃>2 ПРИ подписи — ТРИПАРТИТНЫЙ ШОВ ПЕРЕСЕЧЁН упорядоченной парой (симметрия с C3-S полная)"
    elif exceeds and not sign:
        fork = "(ii) M₃>2 но подпись сломана — новая стена «амплитуда против подписи»"
    else:
        fork = "(iii) M₃≤2 — совместность пары съедена чётностью"
    print(f"  GM-F2j ВЕРДИКТ: развилка {fork}")
    json.dump(dict(M3=M3, M3_order21=M3_21, order_disc=order_disc, pair_max=pair_max,
                   tri_min=tri_min, marg_dev=marg_dev, ns_worst=ns_worst,
                   signature=bool(sign), exceeds_bound=bool(exceeds), ns_ok=bool(ns_ok),
                   fork=fork, sigma=SIG, N=N),
              open(os.path.join(RES, "C4GM_F2j.json"), "w"), indent=2)
    print(f"  → {RES}/C4GM_F2j.json")


if __name__ == "__main__":
    main()
