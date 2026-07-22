"""C4-GHZ / G-T2 (б) m=4 sampling-батарея (prereg v2 449b4ca). Полный блок m=N=4
(цепь взаимного предшествования) реализует алгебр.-максимальный parity-бокс:
s₁s₂s₃s₄ = sign(c(settings)) ⇒ |MK|=Σ|c|=4.000 (>квант 2.828, класс-M NS-бокс).
Подтверждает карандаш числом; NS/CRN-невидимость порядка блока. numpy N=2e6.
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


def swap(op):
    return {tuple(1 - x for x in k): c for k, c in op.items()}


def MK(n):
    op = {(0,): 1.0}
    for _ in range(2, n + 1):
        opp = swap(op); new = {}
        for k, c in op.items():
            new[k + (0,)] = new.get(k + (0,), 0) + 0.5 * c
            new[k + (1,)] = new.get(k + (1,), 0) + 0.5 * c
        for k, c in opp.items():
            new[k + (0,)] = new.get(k + (0,), 0) + 0.5 * c
            new[k + (1,)] = new.get(k + (1,), 0) - 0.5 * c
        op = new
    return op


OP = MK(NP)
SETTINGS = list(itertools.product((0, 1), repeat=NP))
CSIGN = {s: (1 if OP.get(s, 0) >= 0 else -1) for s in SETTINGS}   # sign(c(settings))
ALG = sum(abs(OP.get(s, 0)) for s in SETTINGS)                     # Σ|c| = алгебр. максимум


def block_outcomes(setting, U, last=3):
    """Полный блок: s_free (3 партии) ~½ из U; s_last = target·Πs_free (парити-бокс).
    Порядок блока задаётся `last` (кто разрешается последним) — для CRN-невидимости."""
    parties = [p for p in range(NP) if p != last]
    s = [None] * NP
    for r, p in enumerate(parties):
        s[p] = np.where(U[r] < 0.5, 1, -1)
    prod = np.ones(N, np.int64)
    for p in parties:
        prod = prod * s[p]
    s[last] = CSIGN[setting] * prod                # s₁s₂s₃s₄ = sign(c)
    return s


def E4(setting, U, last=3):
    s = block_outcomes(setting, U, last)
    v = np.ones(N, np.int64)
    for i in range(NP):
        v = v * s[i].astype(np.int64)
    return float(np.mean(v))


def main():
    U = {s: RNG.random((NP - 1, N)) for s in SETTINGS}    # общий поток на настройку (CRN)
    mk = abs(sum(OP[s] * E4(s, U[s]) for s in OP))
    mk_last0 = abs(sum(OP[s] * E4(s, U[s], last=0) for s in OP))   # другой порядок блока
    order_disc = abs(mk - mk_last0)
    # NS: маргинал каждой партии независим от настроек (полный блок, но проверим маргинал ½)
    s0 = block_outcomes(SETTINGS[0], U[SETTINGS[0]])
    marg = [float(np.mean(s0[i] > 0)) for i in range(NP)]
    quantum = 2 ** ((NP - 1) / 2)
    print(f"m=4 sampling (N={N}, σ={SIG:.2e}) — полный блок N=4, парити-бокс:")
    print(f"  |MK| = {mk:.4f}  (карандаш Σ|c|={ALG:.3f}; квант 2^{{(N-1)/2}}={quantum:.3f}; LHV=1)")
    print(f"  порядок блока (last=3 vs last=0) [CRN]: |Δ|={order_disc:.6f}")
    print(f"  маргиналы: {[round(m,4) for m in marg]} (½)")
    beats_q = mk > quantum + 2 * SIG
    ok = abs(mk - ALG) < 3 * SIG and order_disc < 2 * SIG and max(abs(m - 0.5) for m in marg) < 2 * SIG
    print(f"  m=4 ВЕРДИКТ: {'ПРОХОД — |MK|=4.000 (=алгебр., >квант), порядок невидим, маргиналы ½; genuine-N блок сверх-квантовый' if ok else 'разбор'}")
    print(f"  (класс-M полный блок = NS-бокс ⇒ достигает алгебр. 4.000 > квант 2.828; " f"срез до кванта = гипотеза steering-cut, C4_open_problems)")
    json.dump(dict(MK=mk, MK_order2=mk_last0, order_disc=order_disc, algebraic=ALG,
                   quantum=float(quantum), marginals=marg, beats_quantum=bool(beats_q),
                   pass_=bool(ok), sigma=SIG, N=N),
              open(os.path.join(RES, "C4GT2_m4.json"), "w"), indent=2)
    print(f"  → {RES}/C4GT2_m4.json")


if __name__ == "__main__":
    main()
