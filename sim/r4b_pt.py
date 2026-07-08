#!/usr/bin/env python
"""R4b-PT: параллельное темперирование фрустрированного сектора Tw=2π (решение архитектора).

k_t/k_b=0.1 (изгиб-жёсткая), сектора Tw=0 и Tw=2π, точки θ фрустрации+соседи+контроль.
Калибровка лестницы (acceptance горячего конца ≥40%), критерии эквилибрации (§3), сверка
двух лестниц (R=10 vs 14) и двух сидов. Пре-регистрация (a')-(c') — decisions.md.
СВЕРКА С cos² ЗАПРЕЩЕНА до фиксации сырых весов (c').
Результат — results/R4bPT/report.md (инкрементально).
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_pt_fns, classify, branch_counts
from ribbon_sim.frames import sector_sample, total_twist
from ribbon_sim.pt import make_pt, geometric_ladder
from ribbon_sim.experiment import setting_vectors

N = 64
# Сокращённый набор в таймбокс: контроли {0,90} + фрустрация {30,120,172.5}.
THETA_DEG = [0.0, 30.0, 90.0, 120.0, 172.5]
T_COLD = 1e-3
SWAP_EVERY = 50            # чаще обмены ради round-trips (энергошкала k_b=90 требует длинной лестницы)
R_MAIN = 30               # R=10/14 (номинал архитектора) не мостят энергошкалу; нужно ~30
R_ALT = 40               # проверка независимости от лестницы (R_MAIN vs R_ALT)
B_RUN = 1024
N_BLOCKS = 2000
kappa_eq = 1.0; k_e_eq = kappa_eq * (N - 1) * 1.0; total = 3.0 * k_e_eq
r_ratio = 0.1
K_B = total / (2.0 + r_ratio); K_T = r_ratio * K_B
LR = 0.5 / (max(K_B, K_T) + 1.0)
OUT = ROOT / "results" / "R4bPT"; OUT.mkdir(parents=True, exist_ok=True)
rows = []


def cfg_pt():
    return {"lr": LR, "k_e": 0.0, "k_c": 1.0, "spinor": True, "elastic": "cosserat_geo",
            "k_b": K_B, "k_t": K_T, "twist_project": True, "n_twist_corr": 10}


def flush():
    (OUT / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def init_state(key, B, R, tw):
    q0, err = sector_sample(key, (B, N), target=tw)
    return jnp.broadcast_to(q0[:, None], (B, R, N, 4)), err


def run_pt_point(key, theta, tw, R, T_hot, B, n_blocks):
    a, b = setting_vectors(np.radians(theta))
    a = jnp.asarray(a); b = jnp.asarray(b)
    ef, sf = build_pt_fns(cfg_pt(), a, b, tw)
    betas = 1.0 / geometric_ladder(T_hot, T_COLD, R)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    k1, k2 = jax.random.split(key)
    state0, tw_err = init_state(k1, B, R, tw)
    state, diag = run(k2, state0, n_blocks)
    q_cold = state[:, -1]  # холодная реплика = последняя (T_cold)
    s, t = classify(q_cold, a, b)
    cnt = np.asarray(branch_counts(s, t))
    return cnt, diag, float(tw_err)


def calibrate_T_hot(tw, R, B=512, n_blocks=120):
    """Подобрать T_hot: acceptance у горячего конца (звено 0) максимально (цель ≥0.35).
    Энергошкала k_b~90 ⇒ широкая лестница; выбираем T_hot с лучшим min-acceptance."""
    best = (2.0, 0.0)
    for T_hot in [1.5, 2.0, 3.0]:
        cnt, diag, _ = run_pt_point(jax.random.PRNGKey(0), 90.0, tw, R, T_hot, B, n_blocks)
        acc = np.asarray(diag["accept_frac"])
        if acc.min() > best[1]:
            best = (T_hot, float(acc.min()))
    return best


def af_frac(cnt):
    c = cnt.astype(float); return (c[1] + c[2]) / max(c.sum(), 1)


def main():
    rows.append("# R4b-PT: параллельное темперирование фрустрированного Tw=2π\n")
    rows.append(f"k_t/k_b={r_ratio} (изгиб-жёсткая), N={N}, T_cold={T_COLD}, обмен каждые "
                f"{SWAP_EVERY} шагов. Точки θ={THETA_DEG}.\n")
    rows.append("**СВЕРКА С cos²-эталоном ЗАПРЕЩЕНА** (пре-рег c'): сырые веса фиксируются "
                "здесь до любой интерпретации формы.\n")
    t0 = time.time()

    # калибровка лестницы на секторе 2π
    rows.append("## Калибровка лестницы (T_hot: min-acceptance по звеньям, цель ≥0.2 всюду)\n")
    T_hot, acc_min = calibrate_T_hot(2 * np.pi, R=R_MAIN)
    rows.append(f"Выбран T_hot={T_hot}, R_main={R_MAIN} (min-acceptance по звеньям={acc_min:.2f}). "
                f"NB: номинал R=10/14 архитектора не мостит энергошкалу k_b≈90; нужно R≈{R_MAIN}, "
                f"проверка лестницы — R_main vs R_alt={R_ALT}.\n")
    flush(); print(f"T_hot={T_hot} acc_min={acc_min:.2f}")

    # прогон: R_MAIN (2 сида) + R_ALT (сид0) × 2 сектора; сокращённый θ
    results = {}  # (sector, R, seed, theta) -> (cnt, diag, twerr)
    plan = [(R_MAIN, 0), (R_MAIN, 1), (R_ALT, 0)]
    for tw, sec_name in [(0.0, "Tw=0"), (2 * np.pi, "Tw=2π")]:
        for R, seed in plan:
            for th in THETA_DEG:
                key = jax.random.fold_in(jax.random.fold_in(
                    jax.random.PRNGKey(1000 * int(sec_name == "Tw=2π") + R), seed), int(th * 10))
                cnt, diag, twerr = run_pt_point(key, th, tw, R, T_hot, B_RUN, N_BLOCKS)
                results[(sec_name, R, seed, th)] = (cnt, diag, twerr)
            print(f"  {sec_name} R={R} seed={seed} готово ({time.time()-t0:.0f}с)")

    # --- диагностика PT (§3): acceptance, round-trips ---
    rows.append("## Диагностика PT (критерии эквилибрации §3)\n")
    rows.append("| сектор | R | сид | min acc-звено | max acc-звено | min round-trips | Tw-err |")
    rows.append("|---|---|---|---|---|---|---|")
    for (sec, R, seed) in sorted({(k[0], k[1], k[2]) for k in results}):
        accs = [np.asarray(results[(sec, R, seed, th)][1]["accept_frac"]) for th in THETA_DEG]
        rts = [int(results[(sec, R, seed, th)][1]["min_roundtrips"]) for th in THETA_DEG]
        twerrs = [results[(sec, R, seed, th)][2] for th in THETA_DEG]
        amin = min(a.min() for a in accs); amax = max(a.max() for a in accs)
        rows.append(f"| {sec} | {R} | {seed} | {amin:.2f} | {amax:.2f} | {min(rts)} | "
                    f"{max(twerrs):.1e} |")
    flush()

    # --- веса ветвей по θ (сырые, §5) ---
    rows.append("\n## Сырые веса ветвей на T_cold (доля антиферро s·t=−1), R_MAIN\n")
    rows.append("| θ° | Tw=0 сид0 | Tw=0 сид1 | Tw=2π сид0 | Tw=2π сид1 |")
    rows.append("|---|---|---|---|---|")
    for th in THETA_DEG:
        vals = []
        for sec in ["Tw=0", "Tw=2π"]:
            for seed in [0, 1]:
                vals.append(af_frac(results[(sec, R_MAIN, seed, th)][0]))
        rows.append(f"| {th:g} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    flush()

    # --- критерии §3: согласие сидов И лестниц на T_cold ---
    def af(sec, R, seed, th):
        return af_frac(results[(sec, R, seed, th)][0])

    rows.append("\n## Критерии эквилибрации §3 (сектор Tw=2π)\n")
    seed_gap = max(abs(af("Tw=2π", R_MAIN, 0, th) - af("Tw=2π", R_MAIN, 1, th)) for th in THETA_DEG)
    ladder_gap = max(abs(af("Tw=2π", R_MAIN, 0, th) - af("Tw=2π", R_ALT, 0, th)) for th in THETA_DEG)
    acc_all_ok = all(0.2 <= a <= 0.9 for (sec, R, seed, th) in results if sec == "Tw=2π"
                     for a in np.asarray(results[(sec, R, seed, th)][1]["accept_frac"]))
    rt_min = min(int(results[(sec, R, seed, th)][1]["min_roundtrips"])
                 for (sec, R, seed, th) in results if sec == "Tw=2π")
    rows.append(f"- acceptance всех звеньев в [0.2,0.9]: {'✅' if acc_all_ok else '❌'}")
    rows.append(f"- min round-trips (Tw=2π): {rt_min} {'✅' if rt_min >= 3 else '❌ (<3)'}")
    rows.append(f"- согласие 2 сидов на T_cold: max|Δвес|={seed_gap:.3f} {'✅' if seed_gap < 0.05 else '❌'}")
    rows.append(f"- согласие 2 лестниц (R_MAIN vs R_ALT): max|Δвес|={ladder_gap:.3f} {'✅' if ladder_gap < 0.05 else '❌'}")
    equilibrated = acc_all_ok and rt_min >= 3 and seed_gap < 0.05 and ladder_gap < 0.05

    # --- вердикты (a')-(c') ---
    rows.append("\n## Вердикты по пре-регистрации\n")
    rows.append(f"**(a') PT эквилибрирует сектор 2π:** {'✅ ДА (критерии §3 пройдены)' if equilibrated else '❌ НЕТ → фолбэк (iii): неэргодичность как физическая сигнатура сектора'}.")
    af2pi_exist = any(af("Tw=2π", R_MAIN, 0, th) > 0.02 for th in THETA_DEG)
    rows.append(f"\n**(b') веса P(st|θ,2π) на T_cold воспроизводимы И не зависят от лестницы:** "
                f"{'✅' if (seed_gap<0.05 and ladder_gap<0.05) else '❌'} (сид {seed_gap:.3f}, лестница {ladder_gap:.3f}). "
                f"Антиферро-ветвь существует: {'✅' if af2pi_exist else '❌'}.")
    rows.append(f"\n**(c') форма θ-зависимости антиферро — ОТКРЫТА.** Сырые веса зафиксированы "
                "выше. Сверка с cos²/семействами — только после решения архитектора.")
    rows.append(f"\n---\nВремя: {(time.time()-t0)/60:.1f} мин. Таймбокс: 2× стоимости R4b-anneal "
                "(~8.6ч). Если превышен без эквилибрации — фолбэк (iii).")
    flush()
    print(f"R4b-PT готово: equilibrated={equilibrated}, seed_gap={seed_gap:.3f}, ladder_gap={ladder_gap:.3f}")


if __name__ == "__main__":
    main()
