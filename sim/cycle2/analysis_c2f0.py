"""C2-F0 анализ ПОСЛЕ коммита сырья (7f5d176). Ветвление гейта хаоса по
C2F_prereg.md, стадия F0. Только зарегистрированные ветки (i)/(ii)/(iii);
никакой интерпретации за их пределами.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")
S_MIN = 0.5                       # утверждён Артемом 2026-07-16
LADDER = [1e-6, 1e-4, 1e-2]       # точки фита наклона (без π/8, π/2)


def branch_for_cell(cell):
    """Ветвление F0 для одной T-ячейки по prereg. Возврат (verdict, детали)."""
    F = cell["F"]
    def get(d):
        # ближайший ключ по значению δ
        k = min(F.keys(), key=lambda kk: abs(float(kk) - d))
        return F[k]
    ladder = [get(d) for d in LADDER]
    n_zero = sum(1 for r in ladder if r["F"] == 0.0)
    F_1em6 = get(1e-6)["F"]
    F_pi8 = get(np.pi/8)["F"]

    # Ветка (i) ПРОХОД: F(1e−6)=0 событий, ИЛИ [slope≥s_min И F(1e−6)≤0.1·F(π/8)].
    # 0 событий в ≥2 точках из трёх ⇒ проход (пол ниже разрешения).
    if n_zero >= 2:
        return "(i) ПРОХОД", dict(reason=f"{n_zero}/3 точек лестницы = 0 событий "
                                  f"⇒ пол хаоса ниже разрешения M=1200; F(1e−6)={F_1em6}",
                                  slope=None, F_1em6=F_1em6, F_pi8=F_pi8)

    # иначе — фит наклона взвешенным МНК log F vs log δ по ненулевым точкам
    xs, ys, ws = [], [], []
    for d, r in zip(LADDER, ladder):
        if r["F"] > 0:
            xs.append(np.log(d)); ys.append(np.log(r["F"]))
            ws.append((r["F"] / max(r["sigma"], 1e-12)))    # 1/σ_logF = F/σ_F
    slope = None
    if len(xs) >= 2:
        xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)
        W = ws**2
        A = np.vstack([xs, np.ones_like(xs)]).T
        # взвешенный МНК
        sol = np.linalg.lstsq(A * W[:, None], ys * W, rcond=None)[0]
        slope = float(sol[0])
    passes_i = (slope is not None and slope >= S_MIN and F_1em6 <= 0.1 * F_pi8)
    if passes_i:
        return "(i) ПРОХОД", dict(reason=f"slope={slope:.2f}≥{S_MIN} И F(1e−6)≤0.1·F(π/8)",
                                  slope=slope, F_1em6=F_1em6, F_pi8=F_pi8)

    # Ветка (ii) ПЛАТО: F(1e−6),F(1e−4) в пределах 2σ от F(1e−2), при НЕНУЛЕВОМ поле.
    f6, f4, f2 = get(1e-6), get(1e-4), get(1e-2)
    plateau = (abs(f6["F"] - f2["F"]) <= 2*np.hypot(f6["sigma"], f2["sigma"]) and
               abs(f4["F"] - f2["F"]) <= 2*np.hypot(f4["sigma"], f2["sigma"]) and
               f2["F"] > 0)
    if plateau:
        return "(ii) ПЛАТО", dict(reason=f"F(1e−6..1e−2)≈{f2['F']:.3f} в пределах 2σ (ненулевой пол)",
                                  slope=slope, F_1em6=F_1em6, F_pi8=F_pi8)

    return "(iii) СЕРАЯ ЗОНА", dict(reason="ни проход, ни плато — стоп, решение Артема",
                                    slope=slope, F_1em6=F_1em6, F_pi8=F_pi8)


def main():
    d = json.load(open(os.path.join(RES, "C2F0_raw.json")))
    out = {"meta": dict(s_min=S_MIN, raw_commit="7f5d176",
                        prereg=d["meta"]["prereg_commit"]), "cells": {}}
    print(f"=== C2-F0 ветвление (s_min={S_MIN}, сырьё 7f5d176) ===")
    for tc, cell in d["cells"].items():
        assert cell["null_ok"], f"{tc}: НУЛЬ-ТЕСТ γ=0 провален — стоп"
        verdict, det = branch_for_cell(cell)
        # диапазон F по всей лестнице (вкл. π/8, π/2) — диагностика, не гейт
        Fmax = max(r["F"] for r in cell["F"].values())
        Dmax = max(r["Delta"] for r in cell["F"].values())
        out["cells"][tc] = dict(verdict=verdict, F_max_ladder=Fmax, Delta_max=Dmax, **det)
        print(f"  {tc}: {verdict}  [{det['reason']}]")
        print(f"       F_max по всей лестнице (вкл π/8,π/2) = {Fmax:.4f}; Δ_max = {Dmax:.4f}")
    json.dump(out, open(os.path.join(RES, "C2F0_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2F0_analysis.json")


if __name__ == "__main__":
    main()
