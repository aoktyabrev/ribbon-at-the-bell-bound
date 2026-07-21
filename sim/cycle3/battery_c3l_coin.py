"""C3-L / β-монета (prereg — C3L_coin_prereg.md). Механизм: per-реплика монета
w∈{A-first,B-first} (общая, приготовительная, M4′); разрешение цепным правилом
хорды в порядке w. Проверка T3′ бесфреймовой ветви: S=2√2 без сигналинга,
«кто первый» невидим. numpy N=2e6.
"""
import json
import os

import numpy as np

N = 2_000_000
RNG = np.random.default_rng(20260721)
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
ANG = dict(a=0.0, ap=90.0, b=45.0, bp=135.0)


def vec(d):
    t = np.radians(d); return np.array([np.sin(t), 0.0, np.cos(t)])


def resolve(da, db, w, U1, U2):
    """Цепное правило хорды в порядке w (w=True: A-first). Общие U1,U2 (CRN)."""
    c = float(vec(da) @ vec(db))
    s = np.empty(N, np.int8); t = np.empty(N, np.int8)
    A = w  # маска A-first
    # A-first: s из маргинала (½), t по условному P(t=+|s)=(1−s t?...): P(t=+|s)=(1−s·(+1)·... )
    sA = np.where(U1 < 0.5, 1, -1)
    ptp = np.where(sA > 0, (1 - c) / 2, (1 + c) / 2)
    tA = np.where(U2 < ptp, 1, -1)
    # B-first: t из маргинала, s по условному
    tB = np.where(U1 < 0.5, 1, -1)
    psp = np.where(tB > 0, (1 - c) / 2, (1 + c) / 2)
    sB = np.where(U2 < psp, 1, -1)
    s = np.where(A, sA, sB); t = np.where(A, tA, tB)
    return s, t


def E_of(da, db, w, U1, U2):
    s, t = resolve(da, db, w, U1, U2)
    return float(np.mean(s * t)), float(np.mean(s > 0)), float(np.mean(t > 0))


def main():
    # общая приготовительная монета (настройко-независимая) + общий U-поток (CRN)
    w = RNG.random(N) < 0.5
    pairs = {"ab": (ANG["a"], ANG["b"]), "abp": (ANG["a"], ANG["bp"]),
             "apb": (ANG["ap"], ANG["b"]), "apbp": (ANG["ap"], ANG["bp"])}
    U = {k: (RNG.random(N), RNG.random(N)) for k in pairs}
    E = {}; mA = {}; mB = {}
    for k, (da, db) in pairs.items():
        E[k], mA[k], mB[k] = E_of(da, db, w, U[k][0], U[k][1])
    S = abs(E["ab"] - E["abp"] + E["apb"] + E["apbp"])
    tsir = 2 * np.sqrt(2)
    # телеграф: маргинал Боба (t=+) по настройке Алисы (a vs a') при b фикс
    tele = abs(E_of(ANG["a"], ANG["b"], w, *U["ab"])[2]
               - E_of(ANG["ap"], ANG["b"], w, *U["apb"])[2])
    # w-кондиционированные корреляторы: E|w=A vs E|w=B на (a,b)
    sAB, tAB = resolve(ANG["a"], ANG["b"], w, *U["ab"])
    E_wA = float(np.mean((sAB * tAB)[w])); E_wB = float(np.mean((sAB * tAB)[~w]))
    w_invis = abs(E_wA - E_wB)
    print(f"β-монета (N={N}, σ={SIG:.2e}):")
    print(f"  E(a,b)={E['ab']:.4f} (−cos45=−0.7071)  S={S:.4f} (2√2={tsir:.4f}, |Δ|={abs(S-tsir):.4f})")
    print(f"  маргиналы m_A(ab)={mA['ab']:.4f} m_B(ab)={mB['ab']:.4f} (½)")
    print(f"  Δ-телеграф (Боб по настройке Алисы)={tele:.6f} (<2σ={2*SIG:.4f}: {tele<2*SIG})")
    print(f"  w-невидимость |E|w=A − E|w=B|={w_invis:.6f} (<2σ: {w_invis<2*SIG})")
    checks = dict(
        S_tsirelson=bool(abs(S - tsir) < 2 * SIG),
        E_cos=bool(abs(E["ab"] - (-np.cos(np.radians(45)))) < 2 * SIG),
        marginals_half=bool(abs(mA["ab"] - 0.5) < 2 * SIG and abs(mB["ab"] - 0.5) < 2 * SIG),
        telegraph_zero=bool(tele < 2 * SIG),
        w_invisible=bool(w_invis < 2 * SIG))
    allpass = all(checks.values())
    print(f"  ВЕРДИКТ β-монета: {'ПРОХОД ЦЕЛИКОМ (S=2√2 без сигналинга, w невидим)' if allpass else 'ПРОВАЛ: '+str(checks)}")
    json.dump(dict(S=S, tsirelson=float(tsir), E=E, mA=mA, mB=mB, telegraph=tele,
                   w_invisibility=w_invis, checks=checks, all_pass=bool(allpass), sigma=SIG, N=N),
              open(os.path.join(RES, "C3L_coin.json"), "w"), indent=2)
    print(f"  → {RES}/C3L_coin.json")


if __name__ == "__main__":
    main()
