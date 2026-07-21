"""C4-GHZ / G-M GM-F2 J-батарея (prereg 7bfca96). Секторная тренога: λ=(σ₁,σ₂,σ₃)
с сохранённой чётностью σ₁σ₂σ₃=+1 (⊂λ, A1); отклик sᵢ=σᵢ. Ожидание (A4): GHZ-подпись
(попарные 0, тройной ±1) при M₃=2 (LHV-бонд) = стена «трипартитный шов». numpy N=2e6.
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


def sample_lambda():
    """λ=(σ₁,σ₂,σ₃), σ₁σ₂σ₃=+1 (4 конфигурации равновероятны)."""
    s1 = np.where(RNG.random(N) < 0.5, 1, -1)
    s2 = np.where(RNG.random(N) < 0.5, 1, -1)
    s3 = s1 * s2                     # чётность +1 ⇒ σ₃=σ₁σ₂
    return np.stack([s1, s2, s3])    # (3,N)


def response(sig, a):
    """sᵢ(aᵢ,λ)=σᵢ (X и Y — детерм. σᵢ; секторная связка)."""
    return sig                       # a не меняет (закон GM-F2 канон)


def outcomes(sig, a):
    return np.stack([response(sig[i], a[i]) for i in range(3)])


def corr(s, idx):
    v = np.ones(N, np.int64)
    for i in idx:
        v = v * s[i].astype(np.int64)
    return float(np.mean(v))


def main():
    sig = sample_lambda()
    # --- Мермин ---
    M3 = sum(sgn * corr(outcomes(sig, a), (0, 1, 2)) for a, sgn in MERMIN)
    # --- GHZ-подпись: попарные + тройной на всех 8 ---
    pair_max = 0.0; tri_min = 1.0
    for a in SETTINGS:
        s = outcomes(sig, a)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            pair_max = max(pair_max, abs(corr(s, (i, j))))
        tri_min = min(tri_min, abs(corr(s, (0, 1, 2))))
    # --- NS: маргиналы/двухчастичные независимы от третьей настройки ---
    def m(i, a):
        return float(np.mean(outcomes(sig, a)[i] > 0))
    ns1 = max(abs(m(0, ("X", "X", "X")) - m(0, ("X", a2, a3))) for a2 in "XY" for a3 in "XY")
    def e2(i, j, a):
        s = outcomes(sig, a); return corr(s, (i, j))
    ns12 = max(abs(e2(0, 1, ("X", "X", "X")) - e2(0, 1, ("X", "X", a3))) for a3 in "XY")
    ns_worst = max(ns1, ns12)
    # --- моногамия (NS-агностичная): Σ попарных² ≤ 1 (бонд для no-signaling троек) ---
    a0 = ("X", "X", "X"); s0 = outcomes(sig, a0)
    mono_sum = sum(corr(s0, ij) ** 2 for ij in [(0, 1), (0, 2), (1, 2)])
    # CKW QM-reference (GHZ): танглы попарные=0, τ_1(23)=1 ⇒ Σ=0 (контроль)
    print(f"GM-F2 J-батарея (N={N}, σ={SIG:.2e}) — секторная тренога, чётность ⊂λ:")
    print(f"  J-Мермин: M₃ = {M3:.4f} (классич. бонд 2; алгебр. max 4)")
    print(f"  J-GHZ: max|попарные|={pair_max:.6f} (<2σ={2*SIG:.4f}: {pair_max<2*SIG}); "
          f"min|тройной|={tri_min:.4f} (максимал.=1)")
    print(f"  J-NS: worst |Δ маргинал от чужой настройки| = {ns_worst:.6f} (<2σ: {ns_worst<2*SIG})")
    print(f"  J-моногамия (NS): Σ попарных² = {mono_sum:.6f} (≤1 бонд; CKW-ref Σ=0)")
    signature = (pair_max < 2 * SIG) and (tri_min > 1 - 2 * SIG)
    at_bound = abs(abs(M3) - 2.0) < 2 * SIG
    exceeds = abs(M3) > 2.0 + 2 * SIG
    print(f"  GHZ-подпись (0 попарные ∧ ±1 тройной): {signature}")
    print(f"  M₃ на бонде 2: {at_bound}; M₃>2: {exceeds}")
    if signature and at_bound:
        verdict = "СТЕНА «ТРИПАРТИТНЫЙ ШОВ»: GHZ-подпись при M₃=2 (подпись без амплитуды) — прогноз 0.45"
    elif exceeds:
        verdict = "СТОП-АУДИТ: M₃>2 без упоряд. пар = КОНФЛИКТ с T1‴ (⊂λ=LHV) — ловим БАГ, не физику"
    else:
        verdict = "разбор"
    print(f"  GM-F2 ВЕРДИКТ: {verdict}")
    json.dump(dict(M3=M3, pair_max=pair_max, tri_min=tri_min, ns_worst=ns_worst,
                   monogamy_sum=mono_sum, ckw_ref=0.0, signature=bool(signature),
                   at_bound=bool(at_bound), exceeds_bound=bool(exceeds), sigma=SIG, N=N,
                   verdict=verdict), open(os.path.join(RES, "C4GM_F2.json"), "w"), indent=2)
    print(f"  → {RES}/C4GM_F2.json")


if __name__ == "__main__":
    main()
