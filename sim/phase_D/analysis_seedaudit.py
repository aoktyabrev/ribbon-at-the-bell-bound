"""SEED-AUDIT. Анализ ПОСЛЕ коммита сырья (seedaudit_prereg.md «Порядок»).

s_seed по сидам, r = s_seed/σ_bin, вердикт SA-H1/SA-H2, переоценка плато
(M0/M1/M2 по D2-ext с σ_точки = max(σ_bin, s_seed)), таблица всех повторов фазы D.
Функция aicc() перенесена из analysis_ext.py БЕЗ изменений (prereg: формула не меняется).
"""
import json
import os

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

# --- существующие повторы той же конфигурации (нечёт, θ=0, k_f×1, M=1200) ---
# S1-runs: три подэксперимента НОМИНАЛЬНО одной конфигурации (N=32), различие — только split-ключ
S1_N32 = {"R1 kf1.0": 0.41833333333333333, "R2 T0.050": 0.39666666666666667,
          "R3 odd": 0.345}
S1_SIGMA = {"R1 kf1.0": 0.02622017706243009, "R2 T0.050": 0.026499301178766258,
            "R3 odd": 0.027095125637895342}
# DS2 кросс-скан k_f×1 (независимая кампания, свои сиды) — DS2_report §1
DS2_KF1 = {16: 0.387, 32: 0.365, 64: 0.370, 96: 0.377}
# audit_D2_overlap.md: контрольный прогон N=16, PRNGKey(11111111), M=1200
CONTROL_N16 = (0.3533, 0.0270)


def aicc(chi2, k, n):
    """Перенесено из analysis_ext.py::aicc без изменений."""
    aic = chi2 + 2 * k
    denom = n - k - 1
    return aic + (2 * k * (k + 1) / denom if denom > 0 else np.inf)


def spread(vals, sigmas, label):
    v = np.asarray(vals, float)
    s_seed = float(np.std(v, ddof=1))
    sig_bin = float(np.median(sigmas))
    r = s_seed / sig_bin
    # chi2 вокруг общего среднего при биномиальных σ — проверка совместимости
    mu = float(np.mean(v))
    chi2 = float(np.sum(((v - mu) / np.asarray(sigmas, float)) ** 2))
    dof = len(v) - 1
    print(f"\n--- {label}: n={len(v)}")
    print(f"    значения: {', '.join(f'{x:.4f}' for x in sorted(v))}")
    print(f"    mean={mu:.4f}  s_seed={s_seed:.4f} (ddof=1)  σ_bin(медиана)={sig_bin:.4f}")
    print(f"    r = s_seed/σ_bin = {r:.2f}   χ²={chi2:.2f}/{dof} dof")
    return dict(n=len(v), mean=mu, s_seed=s_seed, sigma_bin=sig_bin, r=r,
                chi2=chi2, dof=dof, values=sorted(v.tolist()))


def refit_plateau(sigma_point, tag):
    with open(os.path.join(RES, "D2ext_raw_tables.json")) as f:
        d = json.load(f)
    NS = np.array(d["meta"]["NS"], float)
    A = np.array([abs(d["odd"][f"N{int(n)}|odd|0"]["E_raw"]) for n in NS])
    sA = np.full_like(A, sigma_point)
    n = len(NS)

    w = 1.0 / sA ** 2
    A0 = float(np.sum(w * A) / np.sum(w))
    chi2_0 = float(np.sum(((A - A0) / sA) ** 2))
    aicc0 = aicc(chi2_0, 1, n)

    def m1(N, Aa, g): return Aa * N ** (-g)
    p1, _ = curve_fit(m1, NS, A, p0=[1.0, 0.2], sigma=sA, absolute_sigma=True, maxfev=10000)
    chi2_1 = float(np.sum(((A - m1(NS, *p1)) / sA) ** 2))
    aicc1 = aicc(chi2_1, 2, n)

    def m2(N, Ainf, c, g): return Ainf + c * N ** (-g)
    try:
        p2, _ = curve_fit(m2, NS, A, p0=[0.3, 1.0, 0.5], sigma=sA,
                          absolute_sigma=True, maxfev=20000)
        chi2_2 = float(np.sum(((A - m2(NS, *p2)) / sA) ** 2))
        aicc2 = aicc(chi2_2, 3, n)
    except Exception:
        p2, chi2_2, aicc2 = None, float("nan"), float("inf")

    print(f"\n--- Переоценка плато ({tag}): σ_точки={sigma_point:.4f}")
    print(f"    M0 const : A∞={A0:.4f}  χ²={chi2_0:.3f}  AICc={aicc0:.2f}")
    print(f"    M1 A·N^−γ: A={p1[0]:.3f} γ={p1[1]:+.4f}  χ²={chi2_1:.3f}  AICc={aicc1:.2f}")
    if p2 is not None:
        print(f"    M2 A∞+cN^−γ: A∞={p2[0]:.3f}  χ²={chi2_2:.3f}  AICc={aicc2:.2f}")
    print(f"    ΔAICc(M1−M0) = {aicc1 - aicc0:+.2f}   ΔAICc(M2−M0) = {aicc2 - aicc0:+.2f}")
    return dict(sigma_point=sigma_point, A0=A0, chi2_0=chi2_0, aicc0=aicc0,
                gamma=float(p1[1]), chi2_1=chi2_1, aicc1=aicc1, aicc2=float(aicc2),
                dAICc_M1_M0=float(aicc1 - aicc0), dAICc_M2_M0=float(aicc2 - aicc0))


def main():
    with open(os.path.join(RES, "seedaudit_raw.json")) as f:
        sa = json.load(f)

    print("=" * 72)
    print("SEED-AUDIT — анализ (сырьё закоммичено; prereg: seedaudit_prereg.md)")
    print("=" * 72)

    core = sa["SA_a_core"]
    edge = sa["SA_b_edge"]
    new_core = [(v["seed"], v["A"], v["sigma"]) for v in core.values()]
    new_edge = [(v["seed"], v["A"], v["sigma"]) for v in edge.values()]

    print("\nНОВЫЕ ТОЧКИ (свежие базовые ключи):")
    for sd, a, s in new_core:
        print(f"    core N=32 seed={sd:<9d} A={a:.4f} ± {s:.4f}")
    for sd, a, s in new_edge:
        print(f"    edge N=96 seed={sd:<9d} A={a:.4f} ± {s:.4f}")

    # --- SA-a: ядро, 11 точек = 8 новых + 3 S1 (prereg «Метрики анализа») ---
    core_vals = [a for _, a, _ in new_core] + list(S1_N32.values())
    core_sigs = [s for _, _, s in new_core] + list(S1_SIGMA.values())
    res_core = spread(core_vals, core_sigs, "SA-a ЯДРО N=32 (8 свежих + 3 S1)")

    # --- SA-b: край, 5 точек = 4 новых + D2-ext N=96 ---
    with open(os.path.join(RES, "D2ext_raw_tables.json")) as f:
        d2e = json.load(f)
    d2e_n96 = d2e["odd"]["N96|odd|0"]
    edge_vals = [a for _, a, _ in new_edge] + [abs(d2e_n96["E_raw"])]
    edge_sigs = [s for _, _, s in new_edge] + [d2e_n96["sigma"]]
    res_edge = spread(edge_vals, edge_sigs, "SA-b КРАЙ N=96 (4 свежих + D2-ext)")

    # --- вердикты PREREG ---
    r = res_core["r"]
    print("\n" + "=" * 72)
    print(f"SA-H1: r = {r:.2f}  (порог 1.3)")
    if r <= 1.3:
        print("  ⇒ SA-H1 ПОДТВЕРЖДЁН: разброс совместим с биномиальным; σ фазы D honest.")
        scenario = "A"
    else:
        print("  ⇒ СЦЕНАРИЙ B: r > 1.3, σ_точки := s_seed по всей фазе D, пересчёт ΔAICc.")
        scenario = "B"
    ratio = res_edge["s_seed"] / res_core["s_seed"]
    print(f"SA-H2: s_seed(96)/s_seed(32) = {ratio:.2f}  (порог 1.5)")
    print("  ⇒ SA-H2 " + ("ПОДТВЕРЖДЁН: разброс не растёт с N."
                          if ratio <= 1.5 else "ОПРОВЕРГНУТ: разброс растёт с N."))

    # --- переоценка плато ---
    base = refit_plateau(float(np.median(core_sigs)), "как в D2-ext: σ_bin")
    honest_sigma = max(float(np.median(core_sigs)), res_core["s_seed"])
    seedh = refit_plateau(honest_sigma, "seed-honest: max(σ_bin, s_seed)")

    d = seedh["dAICc_M1_M0"]
    print("\n" + "=" * 72)
    if scenario == "B":
        if d >= 4:
            print(f"  B1: ΔAICc(M1−M0) = {d:+.2f} ≥ 4 — плато СТОИТ при честных ошибках.")
        else:
            print(f"  B2: ΔAICc(M1−M0) = {d:+.2f} < 4 — формулировка плато СМЯГЧАЕТСЯ "
                  "по PREREG (единой правкой).")

    # --- таблица всех повторов фазы D ---
    print("\n" + "=" * 72)
    print("ВСЕ ИЗВЕСТНЫЕ ПОВТОРЫ ОДНОЙ КОНФИГУРАЦИИ (нечёт, θ=0, k_f×1, M=1200):")
    rows = {}
    for n in [16, 32, 48, 64, 96]:
        vals = []
        key = f"N{n}|odd|0"
        if key in d2e["odd"]:
            vals.append(("D2-ext", abs(d2e["odd"][key]["E_raw"])))
        if n in DS2_KF1:
            vals.append(("DS2", DS2_KF1[n]))
        if n == 16:
            vals.append(("контроль-аудит", CONTROL_N16[0]))
        if n == 32:
            vals += [(f"S1 {k}", v) for k, v in S1_N32.items()]
            vals += [(f"seed-audit {sd}", a) for sd, a, _ in new_core]
        if n == 96:
            vals += [(f"seed-audit {sd}", a) for sd, a, _ in new_edge]
        v = np.array([x for _, x in vals], float)
        ss = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        rows[n] = dict(k=len(v), mean=float(v.mean()), s_seed=ss,
                       r=(ss / 0.027 if len(v) > 1 else float("nan")),
                       sources=[s for s, _ in vals], values=v.tolist())
        print(f"  N={n:3d}: k={len(v):2d}  mean={v.mean():.4f}  "
              f"s_seed={ss:.4f}  r={ss/0.027:.2f}" if len(v) > 1 else
              f"  N={n:3d}: k={len(v):2d}  (одна точка)")

    out = dict(prereg="seedaudit_prereg.md", raw_commit_base=sa["meta"]["commit_base"],
               core=res_core, edge=res_edge, scenario=scenario,
               SA_H1_pass=bool(r <= 1.3), SA_H2_pass=bool(ratio <= 1.5),
               ratio_edge_core=ratio, refit_binomial=base, refit_seed_honest=seedh,
               repeats=rows)
    with open(os.path.join(RES, "seedaudit_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → {RES}/seedaudit_analysis.json")


if __name__ == "__main__":
    main()
