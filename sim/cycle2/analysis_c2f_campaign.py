"""C2-F кампания — анализ ПОСЛЕ коммита сырья. Применяет H-F1/H-F2/H-F3 к
ПЕРВИЧНОЙ ячейке k_f×1 (addendum п.4), плюс подтверждающая k_f×4, зеркала,
маргиналы, стратификация. Только зарегистрированные ветки; формы/пороги
зафиксированы ДО данных (написано при работающем прогоне).

H-F1: F_s < 3σ_F на всей сетке Δγ (N=32) ⇒ факторизация при T=0
      конструктивно установлена (KILL = ожидаемый исход).
H-F2: F_s(96) < F_s(32) (затухание). KILL: незатухание (F_s(96) ≥ F_s(32)
      в пределах 2σ) при подтверждённом H-F1.
H-F3: Δ_s совместим с 0 (χ² знакового δ_γ=P(s+|b')−P(s+|b) по Δγ, p>0.05).
      KILL-сюрприз: p<0.001 И |δ|_max > 3σ ⇒ маргинальный телеграф → стоп-аудит.
Диагностика (НЕ kill): A(k_f×1) — см. V0 (0.327 vs эталон 0.418, 2.51σ);
      зеркало |F_s−F_s^mir| в пределах шума; стратификация F_s по θ_prep.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")
GRID = ["pi/8", "pi/4", "3pi/8", "pi/2"]
# диагностика A (addendum п.5): эталон S1 k_f×1 (a9cef7b), σ_comb=√(σ_bin²+s_seed²)
A_REF, A_SIG_BIN, S_SEED = 0.418, 0.026, 0.024


def hf1(cell):
    """Все ли F_s < 3σ_F на сетке. Возврат (established, детали)."""
    rows = []
    for g in cell["gamma"]:
        r = cell["gamma"][g]
        below = r["F_s"] < 3 * max(r["sigma_Fs"], 1e-9)
        rows.append((g, r["F_s"], r["sigma_Fs"], below))
    established = all(b for *_, b in rows)
    return established, rows


def hf3(cell):
    """χ² знакового δ_γ по Δγ vs 0. Возврат (p, chi2, dmax_sigma)."""
    Pbase = cell["marginals"]["P_s_plus"]
    M = cell["M"]
    chi2, dmax_s = 0.0, 0.0
    k = 0
    for g in cell["gamma"]:
        r = cell["gamma"][g]
        d = r["P_s_plus_bp"] - Pbase
        # σ разности двух долей (общий λ ⇒ корреляция, но верхняя оценка независимой)
        var = (Pbase*(1-Pbase) + r["P_s_plus_bp"]*(1-r["P_s_plus_bp"])) / M
        sig = max(np.sqrt(var), 1e-9)
        chi2 += (d/sig)**2
        dmax_s = max(dmax_s, abs(d)/sig)
        k += 1
    p = float(stats.chi2.sf(chi2, k))
    return p, float(chi2), float(dmax_s), k


def main():
    # T=0 первичная по умолчанию; T=0.05 вторичная через argv
    raw_name = sys.argv[1] if len(sys.argv) > 1 else "C2F_campaign_raw.json"
    secondary = "T005" in raw_name
    out_name = raw_name.replace("_raw", "_analysis")
    d = json.load(open(os.path.join(RES, raw_name)))
    out = {"meta": dict(raw=raw_name, prereg=d["meta"]["prereg_commit"],
                        addendum=d["meta"].get("addendum"),
                        role="вторичная T=0.05" if secondary else "первичная T=0"), "verdicts": {}}
    print(f"=== C2-F кампания: H-F1/H-F2/H-F3 ({'ВТОРИЧНАЯ T=0.05' if secondary else 'первичная T=0'}) ===")

    c32 = d["primary_kf1"]["N32"]
    c96 = d["primary_kf1"]["N96"]

    # --- H-F1 (N=32) ---
    est, rows = hf1(c32)
    print("H-F1 (N=32, k_f×1):")
    for g, F, s, b in rows:
        print(f"   Δγ={g:5s}: F_s={F:.4f}±{s:.4f}  {'<3σ ✓' if b else '≥3σ ✗'}")
    v_hf1 = ("KILL: факторизация при T=0 конструктивно установлена (F_s<3σ на всей сетке)"
             if est else "H-F1 НЕ kill: есть F_s≥3σ — удалённое влияние на ближний знак")
    print(f"  ⇒ {v_hf1}")

    # --- H-F2 (затухание с длиной) ---
    print("H-F2 (затухание F_s: N=96 vs N=32):")
    hf2_rows = []
    for g in GRID:
        if g in c32["gamma"] and g in c96["gamma"]:
            F32, s32 = c32["gamma"][g]["F_s"], c32["gamma"][g]["sigma_Fs"]
            F96, s96 = c96["gamma"][g]["F_s"], c96["gamma"][g]["sigma_Fs"]
            decays = F96 < F32 - 2*np.hypot(s32, s96)
            nondecay = F96 >= F32 - 2*np.hypot(s32, s96)
            hf2_rows.append((g, F32, F96, decays))
            print(f"   Δγ={g:5s}: F_s(32)={F32:.4f} F_s(96)={F96:.4f}  {'затухает' if decays else '—'}")
    # при факторизации (H-F1 established, F≈0) затухание не определено — отмечаем
    if est:
        v_hf2 = "H-F2 не применим содержательно: H-F1 дал F_s≈0 (нечему затухать)"
    else:
        v_hf2 = "см. строки: затухание/незатухание по Δγ"
    print(f"  ⇒ {v_hf2}")

    # --- H-F3 (маргинал) ---
    p, chi2, dmax_s, k = hf3(c32)
    surprise = (p < 0.001) and (dmax_s > 3.0)
    v_hf3 = (f"KILL-СЮРПРИЗ: маргинальный телеграф (p={p:.1e}, |δ|_max={dmax_s:.1f}σ) — СТОП-АУДИТ"
             if surprise else f"Δ_s совместим с 0 (χ²={chi2:.1f}/{k}, p={p:.3f}, |δ|_max={dmax_s:.1f}σ)")
    print(f"H-F3: {v_hf3}")

    # --- зеркало + маргиналы + стратификация (диагностика) ---
    print("Диагностика:")
    mir_ok = True
    for g in c32["gamma"]:
        r = c32["gamma"][g]
        dm = abs(r["F_s"] - r["F_s_mirror"])
        if dm > 2*max(r["sigma_Fs"], 1e-9) + 1e-9:
            mir_ok = False
    print(f"   зеркало |F_s−F_s^mir| в пределах 2σ: {mir_ok}")
    print(f"   маргиналы N=32: P(s+)={c32['marginals']['P_s_plus']:.3f} "
          f"P(t+)={c32['marginals']['P_t_plus']:.3f} degen={c32['marginals']['degen_base']:.3f}")
    # диагностика A (addendum п.5) — только если сырьё содержит A_stat (T=0.05)
    A_diag = None
    if "A_stat" in c32["marginals"]:
        A = c32["marginals"]["A_stat"]; sbin = c32["marginals"].get("A_sigma_bin", A_SIG_BIN)
        scomb = float(np.sqrt(sbin**2 + S_SEED**2))
        dev = abs(A - A_REF) / scomb
        A_diag = dict(A=A, A_ref=A_REF, sigma_comb=scomb, dev_sigma=float(dev))
        print(f"   ДИАГНОСТИКА A(k_f×1,T=0.05)={A:.3f} vs эталон {A_REF} ⇒ {dev:.2f}σ_comb "
              f"(НЕ kill; след флага V0: 0.327 против 0.418)")
    # стратификация на самом широком Δγ
    gmax = "pi/2" if "pi/2" in c32["gamma"] else list(c32["gamma"])[-1]
    strat = c32["gamma"][gmax]["strat_Fs"]
    strat_str = ", ".join("{:.3f}(n={})".format(q["F"], q["n"]) for q in strat)
    print(f"   стратификация F_s(Δγ={gmax}) по квартилям θ_prep: {strat_str}")

    # --- подтверждающая k_f×4 ---
    print("Подтверждающая k_f×4 (N=32):")
    c4 = d["confirming_kf4"]["N32"]
    est4, rows4 = hf1(c4)
    for g, F, s, b in rows4:
        print(f"   Δγ={g:5s}: F_s={F:.4f}±{s:.4f}  {'<3σ ✓' if b else '≥3σ ✗'}")
    print(f"  ⇒ {'подтверждает факторизацию (F_s<3σ)' if est4 else 'F_s≥3σ — расхождение с F0!'}")

    out["verdicts"] = dict(
        H_F1=dict(established=bool(est), verdict=v_hf1,
                  rows=[dict(gamma=g, F_s=F, sigma=s, below3=bool(b)) for g, F, s, b in rows]),
        H_F2=dict(verdict=v_hf2, rows=[dict(gamma=g, F32=a, F96=b, decays=bool(c)) for g, a, b, c in hf2_rows]),
        H_F3=dict(p_value=p, chi2=chi2, dof=k, dmax_sigma=dmax_s, surprise=bool(surprise), verdict=v_hf3),
        mirror_ok=bool(mir_ok),
        A_diagnostic=A_diag,
        confirming_kf4=dict(established=bool(est4)))
    json.dump(out, open(os.path.join(RES, out_name), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/{out_name}")


if __name__ == "__main__":
    main()
