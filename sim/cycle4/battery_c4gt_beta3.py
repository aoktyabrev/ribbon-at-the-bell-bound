"""C4-GHZ / G-T β3 (addendum2 840602b). ДЫРА-N2: кандидат N=4 = одна PR-пара (1,2)
+ локальный «45°-хвост» x_j=y_j=1 (детерм.). Карандаш: кандидат ТИЕТ бонд (M₄=±4),
не бьёт. β3 подтверждает численно + NS + порядко-невидимость. Конвенция Мермина
выписана до прогона (Re нечёт / Im чёт; N=4 → Im; LHV-бонд 4). numpy N=2e6.
"""
import itertools
import json
import os

import numpy as np

N = 2_000_000
NP = 4                                    # число партий
RNG = np.random.default_rng(20260721)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
SETTINGS = list(itertools.product("XY", repeat=NP))
LHV_BOUND_N4 = 4                          # выписано карандашом (перебор, addendum2)


def coeff_im(s):                          # N=4 чётное ⇒ Im: #Y нечёт, (−1)^{(#Y−1)/2}
    ny = s.count("Y"); return 0 if ny % 2 == 0 else (-1) ** ((ny - 1) // 2)


def coeff_re(s):                          # контроль
    ny = s.count("Y"); return 0 if ny % 2 else (-1) ** (ny // 2)


def outcomes(a, eps, order="12"):
    a1, a2 = a[0], a[1]
    sig = -1 if (a1 == "Y" and a2 == "Y") else 1
    if order == "12":
        s1 = eps; s2 = eps * sig
    else:
        s2 = eps; s1 = eps * sig
    tail = [np.ones(N, np.int8) for _ in range(NP - 2)]   # x_j=y_j=1 (детерм.)
    return [s1, s2] + tail


def corr(s, idx):
    v = np.ones(N, np.int64)
    for i in idx:
        v = v * s[i].astype(np.int64)
    return float(np.mean(v))


def M4(eps, cf, order="12"):
    return sum(cf(a) * corr(outcomes(a, eps, order), tuple(range(NP))) for a in SETTINGS)


def main():
    eps = np.where(RNG.random(N) < 0.5, 1, -1)
    m_im = M4(eps, coeff_im); m_re = M4(eps, coeff_re)
    m_im_21 = M4(eps, coeff_im, "21")
    # NS: маргинал party1 независим от чужих настроек; двухчастичный (1,2) от a3,a4
    def m1(a):
        return float(np.mean(outcomes(a, eps)[0] > 0))
    ns1 = max(abs(m1(("X",) * NP) - m1(("X",) + a[1:])) for a in SETTINGS)
    def e12(a):
        s = outcomes(a, eps); return corr(s, (0, 1))
    ns12 = max(abs(e12(("X", "X", "X", "X")) - e12(("X", "X") + a[2:])) for a in SETTINGS)
    ns_worst = max(ns1, ns12)
    order_disc = abs(m_im - m_im_21)
    print(f"β3 G-T N={NP} (N_samp={N}, σ={SIG:.2e}) — одна PR-пара + детерм. хвост:")
    print(f"  конвенция: Im (чётное N=4); LHV-бонд (карандаш) = {LHV_BOUND_N4}")
    print(f"  M₄ (Im) = {m_im:.4f}  |  M₄ (Re, контроль) = {m_re:.4f}  |  order-21 = {m_im_21:.4f}")
    print(f"  |M₄| vs LHV-бонд {LHV_BOUND_N4}: {'БЬЁТ' if abs(m_im) > LHV_BOUND_N4 + 2*SIG else 'ТИЕТ/ниже'}")
    print(f"  NS worst = {ns_worst:.6f} (<2σ={2*SIG:.4f}: {ns_worst<2*SIG})")
    print(f"  порядко-невидимость пары [CRN] = {order_disc:.6f}")
    beats = abs(m_im) > LHV_BOUND_N4 + 2 * SIG
    print(f"  β3 ВЕРДИКТ: конструкция {'БЬЁТ бонд (ДЫРА-N2 закрывается)' if beats else 'ТИЕТ бонд (M₄=±4), НЕ бьёт ⇒ детерм. хвост классичен, √2 не набирается; ДЫРА-N2 ОСТАЁТСЯ ОТКРЫТОЙ'}")
    print(f"  (статус theorem(N=3 value) НЕ откатывается — addendum2)")
    json.dump(dict(M4_im=m_im, M4_re=m_re, M4_im_order21=m_im_21, lhv_bound=LHV_BOUND_N4,
                   beats_bound=bool(beats), ns_worst=ns_worst, order_disc=order_disc,
                   sigma=SIG, N=N, parties=NP,
                   note="candidate ties LHV bound (deterministic tail is classical, no √2); "
                        "ДЫРА-N2 stays open; theorem(N=3 value) intact"),
              open(os.path.join(RES, "C4GT_beta3.json"), "w"), indent=2)
    print(f"  → {RES}/C4GT_beta3.json")


if __name__ == "__main__":
    main()
