"""C4-GHZ / G-T β2 (addendum1 64ed88f). ДЫРА-N1 закрыта усилением: ОДНА
упорядоченная пара (1,2) даёт M₃=4 (алгебр. максимум). Конструкция:
s₁=ε; s₂=ε·σ(a₁,a₂), σ=−1⇔(Y,Y); s₃=+1@X,−1@Y (детерм.). PR-пара + локальная
третья. Проверка: M₃=4, маргиналы, NS (одно/двухчастичные от третьей настройки),
порядко-невидимость пары [CRN]. numpy N=2e6.
"""
import itertools
import json
import os

import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260721)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
SETTINGS = list(itertools.product("XY", repeat=3))       # 8 комбо
MERMIN = [(("X", "X", "X"), +1), (("X", "Y", "Y"), -1),
          (("Y", "X", "Y"), -1), (("Y", "Y", "X"), -1)]


def outcomes(a, eps, order="12"):
    """s₁,s₂,s₃ по конструкции addendum1. order: '12' (1→2) или '21' (2→1), CRN eps."""
    a1, a2, a3 = a
    sig = -1 if (a1 == "Y" and a2 == "Y") else 1
    if order == "12":
        s1 = eps; s2 = eps * sig
    else:                        # 2→1: пара упорядочена наоборот, тот же ε (CRN)
        s2 = eps; s1 = eps * sig
    s3 = np.where(np.full(N, a3 == "X"), 1, -1)
    return s1, s2, s3


def E(a, eps, order="12"):
    s1, s2, s3 = outcomes(a, eps, order)
    return float(np.mean(s1 * s2 * s3))


def main():
    eps = np.where(RNG.random(N) < 0.5, 1, -1)           # ε ~ ½ (общий поток, CRN)
    # --- M₃ ---
    M3 = sum(sign * E(a, eps) for a, sign in MERMIN)
    # --- маргиналы одно-частичные на всех 8 комбо ---
    marg = {}
    for a in SETTINGS:
        s1, s2, s3 = outcomes(a, eps)
        marg["".join(a)] = [float(np.mean(s1 > 0)), float(np.mean(s2 > 0)), float(np.mean(s3 > 0))]
    # --- NS: маргинал party i независим от настроек ДРУГИХ ---
    # party3 single от (a1,a2): при фикс a3=X, менять a1,a2
    def m3(a1, a2, a3):
        return float(np.mean(outcomes((a1, a2, a3), eps)[2] > 0))
    ns_p3 = max(abs(m3("X", "X", "X") - m3(a1, a2, "X")) for a1 in "XY" for a2 in "XY")
    def m1(a1, a2, a3):
        return float(np.mean(outcomes((a1, a2, a3), eps)[0] > 0))
    ns_p1 = max(abs(m1("X", "X", "X") - m1("X", a2, a3)) for a2 in "XY" for a3 in "XY")
    # двухчастичный (1,2) от a3
    def e12(a1, a2, a3):
        s1, s2, _ = outcomes((a1, a2, a3), eps); return float(np.mean(s1 * s2))
    ns_12 = max(abs(e12("X", "X", "X") - e12("X", "X", a3)) for a3 in "XY")
    # двухчастичный (1,3) от a2
    def e13(a1, a2, a3):
        s1, _, s3 = outcomes((a1, a2, a3), eps); return float(np.mean(s1 * s3))
    ns_13 = max(abs(e13("X", "X", "X") - e13("X", a2, "X")) for a2 in "XY")
    ns_worst = max(ns_p3, ns_p1, ns_12, ns_13)
    # --- порядко-невидимость пары [CRN]: M₃ при order 12 vs 21 ---
    M3_21 = sum(sign * E(a, eps, "21") for a, sign in MERMIN)
    order_disc = abs(M3 - M3_21)

    print(f"β2 G-T (N={N}, σ={SIG:.2e}) — одна упорядоченная пара (1,2):")
    print(f"  M₃ = {M3:.4f} (алгебр. max 4; order-21 M₃={M3_21:.4f})")
    print(f"  NS worst (одно/двухчастичные от чужой настройки) = {ns_worst:.6f} (<2σ={2*SIG:.4f})")
    print(f"  порядко-невидимость пары [CRN] |M₃(12)−M₃(21)| = {order_disc:.6f}")
    print("  маргиналы одно-частичные (party1/party2/party3) на 8 комбо:")
    for k, v in marg.items():
        print(f"    {k}: {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")
    p12_half = max(abs(v[0] - 0.5) for v in marg.values()) < 2 * SIG and \
        max(abs(v[1] - 0.5) for v in marg.values()) < 2 * SIG
    p3_det = all(abs(v[2] - 0.5) > 0.4 for v in marg.values())
    print(f"  party1&2 маргиналы ½: {p12_half}; party3 ДЕТЕРМИНИРОВАН (±1): {p3_det}")
    print("  ⇒ ОГОВОРКА addendum1 конкретно: M₃=4 достигнут ОДНОЙ парой, NS цел,")
    print("    но party3-маргинал детерминирован ⇒ это M₃-ЗНАЧЕНИЕ-свидетель,")
    print("    НЕ полная GHZ-статистика (½ на всех 8 держится для пары, не для party3).")
    checks = dict(
        M3_is_4=bool(abs(M3 - 4.0) < 2 * SIG),
        NS_intact=bool(ns_worst < 2 * SIG),
        pair_order_invisible=bool(order_disc < 2 * SIG),
        pair_marginals_half=bool(p12_half),
        party3_deterministic=bool(p3_det))
    print(f"  β2 ВЕРДИКТ: M₃=4 {checks['M3_is_4']}, NS {checks['NS_intact']}, "
          f"порядок-невидим {checks['pair_order_invisible']} ⇒ "
          f"{'ПРОХОД (одна пара ⇒ M₃=4, NS, невидимость)' if (checks['M3_is_4'] and checks['NS_intact'] and checks['pair_order_invisible']) else 'разбор'}")
    json.dump(dict(M3=M3, M3_order21=M3_21, order_disc=order_disc, ns_worst=ns_worst,
                   ns=dict(p3=ns_p3, p1=ns_p1, e12=ns_12, e13=ns_13),
                   marginals=marg, checks=checks, sigma=SIG, N=N,
                   note="Mermin-VALUE witness (one ordered pair ⇒ M₃=4); party3 deterministic ⇒ "
                        "not full GHZ statistics (open, per addendum1 oговорка)"),
              open(os.path.join(RES, "C4GT_beta2.json"), "w"), indent=2)
    print(f"  → {RES}/C4GT_beta2.json")


if __name__ == "__main__":
    main()
