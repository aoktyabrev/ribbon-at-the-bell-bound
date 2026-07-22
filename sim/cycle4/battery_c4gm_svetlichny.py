"""C4-GHZ / GM-F2j Свеличны-батарея (prereg-мини C4GM_svetlichny_prereg.md).
S₃=|M₃±M₃'| на конструкции GM-F2j (чётность ⊂λ + упоряд. пара). Прогноз: S₃≤4
(biseparable бонд — фиксированная пара TOBL-ограничена), S₃>4=СТОП-АУДИТ.
Билокальный бонд 4; квант GHZ 4√2. numpy N=2e6.
"""
import itertools
import json
import os

import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260722)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")


def w(a1, a2):
    return 1 if (a1 == "X" and a2 == "X") else -1


def source():
    s1 = np.where(RNG.random(N) < 0.5, 1, -1)
    s2 = np.where(RNG.random(N) < 0.5, 1, -1)
    return s1, s2, s1 * s2                       # чётность +1


def E(sig, a):
    σ1, σ2, σ3 = sig
    s1 = σ1; s2 = σ2 * w(a[0], a[1]); s3 = σ3 * 1
    return float(np.mean(s1.astype(np.int64) * s2 * s3))


def main():
    sig = source()
    Emap = {a: E(sig, a) for a in itertools.product("XY", repeat=3)}
    M3 = Emap[("X", "X", "X")] - Emap[("X", "Y", "Y")] - Emap[("Y", "X", "Y")] - Emap[("Y", "Y", "X")]
    M3p = Emap[("Y", "Y", "Y")] - Emap[("Y", "X", "X")] - Emap[("X", "Y", "X")] - Emap[("X", "X", "Y")]
    S_plus = abs(M3 + M3p); S_minus = abs(M3 - M3p)
    S3 = max(S_plus, S_minus)
    bisep, quantum = 4.0, 4 * np.sqrt(2)
    print(f"GM-F2j Свеличны (N={N}, σ={SIG:.2e}):")
    print(f"  M₃={M3:.4f}  M₃'={M3p:.4f}  ⇒ |M₃+M₃'|={S_plus:.4f}, |M₃−M₃'|={S_minus:.4f}")
    print(f"  S₃ = {S3:.4f}  (biseparable бонд {bisep}; квант GHZ {quantum:.3f})")
    exceeds = S3 > bisep + 2 * SIG
    if exceeds:
        verdict = "СТОП-АУДИТ: S₃>4 без блока размера 3 = КОНФЛИКТ с biseparable/TOBL — БАГ, не физика"
    else:
        verdict = ("S₃≤4 ✓ (biseparable). Уточнённая стена: «пара покупает Мермина (M₃=4), "
                   "но НЕ Свеличны (S₃=4=бонд): genuine-N требует блока размера N» — мост к лестнице размера блока")
    print(f"  ВЕРДИКТ: {verdict}")
    json.dump(dict(M3=M3, M3_prime=M3p, S3=S3, S_plus=S_plus, S_minus=S_minus,
                   biseparable_bound=bisep, quantum_ghz=float(quantum),
                   exceeds=bool(exceeds), verdict=verdict, sigma=SIG, N=N),
              open(os.path.join(RES, "C4GM_svetlichny.json"), "w"), indent=2)
    print(f"  → {RES}/C4GM_svetlichny.json")


if __name__ == "__main__":
    main()
