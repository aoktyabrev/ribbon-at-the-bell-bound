#!/usr/bin/env python
"""R4c — дожатие меры антиферро-сектора Tw=2π (решение архитектора).

Адаптивная лестница (равный acceptance), свопы каждые 10 шагов, полный свип θ (25 точек),
оба сектора, 2 сида, 2 независимые лестницы. Round-trips + ладдер/сид-независимость.
Фиты формы ПОСЛЕ фиксации сырых весов (масштаб.синглет/хорда/ступень/пила, AIC, bootstrap).
Диагностика меры (c_A,c_B по ветвям) на T_cold. Результат — results/R4c/report.md.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim import analysis, plots
from ribbon_sim.dynamics import build_pt_fns, classify, branch_counts
from ribbon_sim.frames import sector_sample, total_twist, axis
from ribbon_sim.pt import make_pt, geometric_ladder
from ribbon_sim.experiment import setting_vectors

N = 64
THETA_DEG = np.arange(0.0, 180.01, 7.5)           # полный свип, 25 точек
MEAS_THETA = [0.0, 60.0, 120.0]                   # θ для диагностики меры
T_COLD = 1e-3
SWAP_EVERY = 10
B_RUN = 512
N_BLOCKS = 6000                                   # 60k шагов (mean-RT≥3 при R≈30, swap10)
kappa_eq = 1.0; total = 3.0 * kappa_eq * (N - 1)
K_B = total / 2.1; K_T = 0.1 * K_B; LR = 0.5 / (max(K_B, K_T) + 1.0)
OUT = ROOT / "results" / "R4c"; OUT.mkdir(parents=True, exist_ok=True)
rows = []


def cfg_pt():
    return {"lr": LR, "k_e": 0.0, "k_c": 1.0, "spinor": True, "elastic": "cosserat_geo",
            "k_b": K_B, "k_t": K_T, "twist_project": True, "n_twist_corr": 10}


def flush():
    (OUT / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def pilot_acc(betas, tw, B=256, n_blocks=200):
    """Средний acceptance по звеньям при данной лестнице (пилот на θ=90)."""
    a, b = setting_vectors(np.radians(90.0)); a = jnp.asarray(a); b = jnp.asarray(b)
    ef, sf = build_pt_fns(cfg_pt(), a, b, tw)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    q0, _ = sector_sample(jax.random.PRNGKey(0), (B, N), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B, len(betas), N, 4))
    _, d = run(jax.random.PRNGKey(1), st0, n_blocks)
    return np.asarray(d["accept_frac"])


def adaptive_ladder(tw, T_hot=1.5, target=0.35, max_R=40):
    """Адаптивная лестница равного acceptance: бисекция низких звеньев до min-acc≥target."""
    Ts = list(np.geomspace(T_hot, T_COLD, 12))      # старт: 12 геом. ступеней
    for _ in range(30):
        betas = 1.0 / jnp.asarray(sorted(Ts, reverse=True))
        acc = pilot_acc(betas, tw)
        if acc.min() >= target or len(Ts) >= max_R:
            break
        j = int(np.argmin(acc))                     # худшее звено
        Ts_sorted = sorted(Ts, reverse=True)
        mid = np.sqrt(Ts_sorted[j] * Ts_sorted[j + 1])  # геом. середина
        Ts = Ts_sorted; Ts.insert(j + 1, mid)
    Ts = sorted(Ts, reverse=True)
    return 1.0 / jnp.asarray(Ts), len(Ts), float(acc.min())


def run_point(key, theta, tw, betas, n_blocks, want_measure=False):
    a, b = setting_vectors(np.radians(theta)); a = jnp.asarray(a); b = jnp.asarray(b)
    ef, sf = build_pt_fns(cfg_pt(), a, b, tw)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    k1, k2 = jax.random.split(key)
    q0, twerr = sector_sample(k1, (B_RUN, N), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B_RUN, len(betas), N, 4))
    state, diag = run(k2, st0, n_blocks)
    qc = state[:, -1]                               # холодная реплика
    s, t = classify(qc, a, b); s = np.asarray(s); t = np.asarray(t)
    cnt = np.asarray(branch_counts(jnp.asarray(s), jnp.asarray(t)))
    meas = None
    if want_measure:
        nA = np.asarray(axis(qc[:, 0])); nB = np.asarray(axis(qc[:, -1]))
        meas = {"c_A": nA @ np.asarray(a), "c_B": nB @ np.asarray(b), "s": s, "t": t}
    return cnt, diag, float(twerr), meas


def af(cnt):
    c = cnt.astype(float); return (c[1] + c[2]) / max(c.sum(), 1)


def E_of(cnt):
    c = cnt.astype(float); n = c.sum()
    return (c[0] + c[3] - c[1] - c[2]) / max(n, 1)


def main():
    rows.append("# R4c — дожатая мера антиферро-сектора Tw=2π\n")
    rows.append(f"k_t/k_b=0.1, N={N}, адаптивная лестница (равный acceptance), свопы/{SWAP_EVERY} "
                f"шагов, {N_BLOCKS} блоков (~{N_BLOCKS*SWAP_EVERY//1000}k шагов), B={B_RUN}, полный свип θ.\n")
    t0 = time.time()

    # адаптивная лестница + вторая независимая (геометрическая) для ладдер-проверки
    rows.append("## Лестницы\n")
    betas_ad, R_ad, accmin_ad = adaptive_ladder(2 * np.pi)
    betas_geo = 1.0 / geometric_ladder(1.5, T_COLD, R_ad)  # вторая: геом. той же длины
    rows.append(f"- Лестница 1 (адаптивная): R={R_ad}, min-acc={accmin_ad:.2f}.")
    rows.append(f"- Лестница 2 (геометрическая, независимая): R={R_ad}.\n")
    flush(); print(f"адаптивная лестница R={R_ad} min-acc={accmin_ad:.2f}")

    ladders = {"adapt": betas_ad, "geom": betas_geo}
    # сетка прогонов: 2 сектора × 2 лестницы × 2 сида × 25 θ
    res = {}   # (sec, ladder, seed, th) -> (cnt, diag, twerr, meas)
    for tw, sec in [(0.0, "Tw0"), (2 * np.pi, "Tw2pi")]:
        for lad_name, betas in ladders.items():
            for seed in [0, 1]:
                for th in THETA_DEG:
                    key = jax.random.PRNGKey(hash((sec, lad_name, seed, round(float(th), 1))) % (2**31))
                    wm = (lad_name == "adapt" and seed == 0 and float(th) in MEAS_THETA and sec == "Tw2pi")
                    res[(sec, lad_name, seed, float(th))] = run_point(key, th, tw, betas, N_BLOCKS, wm)
                print(f"  {sec} {lad_name} seed{seed} готово ({(time.time()-t0)/60:.1f}мин)")
                flush()

    thr = np.radians(THETA_DEG)

    # --- диагностика PT ---
    rows.append("## Диагностика PT (§3)\n")
    rows.append("| сектор | лестница | сид | min-acc | mean-RT | min-RT | Tw-err |")
    rows.append("|---|---|---|---|---|---|---|")
    for (sec, lad, seed) in sorted({(k[0], k[1], k[2]) for k in res}):
        accs = [np.asarray(res[(sec, lad, seed, float(th))][1]["accept_frac"]) for th in THETA_DEG]
        mrt = np.mean([float(res[(sec, lad, seed, float(th))][1]["mean_roundtrips"]) for th in THETA_DEG])
        minrt = min(int(res[(sec, lad, seed, float(th))][1]["min_roundtrips"]) for th in THETA_DEG)
        twe = max(res[(sec, lad, seed, float(th))][2] for th in THETA_DEG)
        rows.append(f"| {sec} | {lad} | {seed} | {min(a.min() for a in accs):.2f} | {mrt:.1f} | {minrt} | {twe:.1e} |")
    flush()

    # --- сырые веса (антиферро-доля) по θ ---
    rows.append("\n## Сырые веса: доля антиферро P(s·t=−1) по θ (адапт. лестница, сид0)\n")
    rows.append("| θ° | Tw=0 | Tw=2π | E_Tw2π |")
    rows.append("|---|---|---|---|")
    for th in THETA_DEG:
        c0 = res[("Tw0", "adapt", 0, float(th))][0]; c2 = res[("Tw2pi", "adapt", 0, float(th))][0]
        rows.append(f"| {th:g} | {af(c0):.3f} | {af(c2):.3f} | {E_of(c2):+.3f} |")
    flush()

    # --- критерии эквилибрации ---
    def afq(sec, lad, seed, th):
        return af(res[(sec, lad, seed, float(th))][0])
    seed_gap = max(abs(afq("Tw2pi", "adapt", 0, th) - afq("Tw2pi", "adapt", 1, th)) for th in THETA_DEG)
    ladder_gap = max(abs(afq("Tw2pi", "adapt", 0, th) - afq("Tw2pi", "geom", 0, th)) for th in THETA_DEG)
    # 2σ порог: σ на долю ~ sqrt(p(1-p)/B), берём худший
    sig2 = 2 * np.sqrt(0.25 / B_RUN)
    mean_rt = np.mean([float(res[("Tw2pi", "adapt", 0, float(th))][1]["mean_roundtrips"]) for th in THETA_DEG])
    rows.append("\n## Критерии эквилибрации §3 (Tw=2π)\n")
    rows.append(f"- mean round-trips: {mean_rt:.1f} {'✅' if mean_rt >= 3 else '❌'} (цель ≥3)")
    rows.append(f"- согласие сидов: max|Δ|={seed_gap:.3f} vs 2σ={sig2:.3f} {'✅' if seed_gap < sig2 else '❌'}")
    rows.append(f"- согласие лестниц: max|Δ|={ladder_gap:.3f} vs 2σ={sig2:.3f} {'✅' if ladder_gap < sig2 else '❌'}")
    equilibrated = mean_rt >= 3 and seed_gap < sig2 and ladder_gap < sig2
    flush()

    # --- ключевой вопрос (a): E(0|2π) ---
    E0_2pi = E_of(res[("Tw2pi", "adapt", 0, 0.0)][0])
    af0_2pi = af(res[("Tw2pi", "adapt", 0, 0.0)][0])
    rows.append(f"\n## (a) Ключевой вопрос: P_анти(θ=0|2π) → 1 или пол?\n")
    rows.append(f"P_анти(0|2π) = **{af0_2pi:.3f}** (E(0)={E0_2pi:+.3f}). "
                + ("→ близко к 1 (полный антиферро)." if af0_2pi > 0.9 else
                   f"→ ВЫХОД НА ПОЛ <1 (ферро-примесь {1-af0_2pi:.2f}): термодинамическая, не транзиент.") )
    flush()

    # --- (c) фиты формы (ПОСЛЕ фиксации весов выше) ---
    E2 = np.array([E_of(res[("Tw2pi", "adapt", 0, float(th))][0]) for th in THETA_DEG])
    ncnt = np.array([res[("Tw2pi", "adapt", 0, float(th))][0].sum() for th in THETA_DEG])
    sigE = np.sqrt(np.maximum(1 - E2**2, 1e-4) / ncnt)
    fits = analysis.fit_shapes(thr, E2, sigE)
    h = analysis.harmonics(thr, E2)
    rows.append("\n## (c) Фиты формы E(θ|2π) [сырые веса зафиксированы выше]\n")
    rows.append("| модель | параметр | χ² | AIC |")
    rows.append("|---|---|---|---|")
    rows.append(f"| масштаб. синглет −A·cosθ | A={fits['singlet']['A']:.3f} | {fits['singlet']['chi2']:.1f} | {fits['singlet']['aic']:.1f} |")
    rows.append(f"| хорда E_p | p={fits['chord_p']['p']:.3f} | {fits['chord_p']['chi2']:.1f} | {fits['chord_p']['aic']:.1f} |")
    rows.append(f"| ступень −A·sign(cosθ) | A={fits['step']['A']:.3f} | {fits['step']['chi2']:.1f} | {fits['step']['aic']:.1f} |")
    rows.append(f"| пила −A(1−2θ/π) | A={fits['saw']['A']:.3f} | {fits['saw']['chi2']:.1f} | {fits['saw']['aic']:.1f} |")
    rows.append(f"\nЛУЧШАЯ по AIC: **{fits['best']}**. Гармоники: A1={h['A1']:+.3f} A3={h['A3']:+.3f} (A3/A1={h['A3']/h['A1'] if abs(h['A1'])>1e-6 else float('nan'):+.3f}).")
    rows.append(f"Первичная сверка (архитектор): знак и нуль@90° как синглет, края задавлены (E(0)={E0_2pi:.2f}) — «ножницы» на уровне меры; на хорде не лежит если E(0)≠−1.\n")
    flush()

    # --- график E(θ) обоих секторов + фит ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    E0s = np.array([E_of(res[("Tw0", "adapt", 0, float(th))][0]) for th in THETA_DEG])
    fig, ax = plt.subplots(figsize=(8, 5))
    g = np.linspace(0, np.pi, 200)
    ax.plot(np.degrees(g), -np.cos(g), "k--", lw=1, alpha=0.5, label="−cosθ (синглет)")
    ax.plot(THETA_DEG, E0s, "C0o-", label="Tw=0 (ферро)")
    ax.plot(THETA_DEG, E2, "C3o-", label="Tw=2π (антиферро)")
    ax.plot(THETA_DEG, fits[fits['best']]['curve'], "C3:", lw=1, label=f"фит {fits['best']}")
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("θ°"); ax.set_ylabel("E(θ)")
    ax.set_title("R4c: E(θ) по секторам Tw"); ax.legend(fontsize=8); ax.grid(alpha=0.2); ax.set_ylim(-1.1, 1.1)
    fig.tight_layout(); fig.savefig(OUT / "E_sectors.png", dpi=120); plt.close(fig)
    rows.append("![E(θ) по секторам](E_sectors.png)\n")

    # --- диагностика меры (c_A,c_B по ветвям) на T_cold, Tw=2π ---
    measure = {}
    for th in MEAS_THETA:
        m = res[("Tw2pi", "adapt", 0, th)][3]
        if m:
            measure[th] = {"c_A": m["c_A"], "c_B": m["c_B"], "s": m["s"], "t": m["t"],
                           "tau": np.zeros(N - 1), "Tw0": 2 * np.pi, "Tw1": 2 * np.pi}
    if measure:
        plots.plot_measure_diag(measure, OUT / "measure_2pi.png", title="R4c мера Tw=2π (c_A,c_B по ветвям)")
        rows.append("## Диагностика меры Tw=2π на T_cold (впервые для антиферро-сектора)\n")
        rows.append("![мера](measure_2pi.png)\n")
        rows.append("Сверка с КС-эталоном ∝ max(c,0) — на графике (оверлей). Какую плотность рождает заряженный сектор.\n")

    # --- вердикты ---
    rows.append("## Вердикты по пре-регистрации\n")
    rows.append(f"**(a)** P_анти(0|2π)={af0_2pi:.3f} — {'полный антиферро' if af0_2pi>0.9 else 'ПОЛ <1 (ферро-примесь термодинамическая)'}.")
    rows.append(f"\n**(b)** мера сектора определена (ладдер+сид <2σ): {'✅' if (seed_gap<sig2 and ladder_gap<sig2) else '❌'} (сид {seed_gap:.3f}, ладдер {ladder_gap:.3f}, 2σ={sig2:.3f}). Эквилибрация: {'✅' if equilibrated else '⚠️ частичная'}.")
    rows.append(f"\n**(c)** форма: лучшая модель **{fits['best']}** (см. фиты). Сверка разблокирована.")
    rows.append(f"\n---\nВремя: {(time.time()-t0)/60:.1f} мин.")
    flush()
    print(f"R4c готово: equil={equilibrated}, E(0|2π)={E0_2pi:.3f}, best_fit={fits['best']}, seed_gap={seed_gap:.3f} ladder_gap={ladder_gap:.3f}")


if __name__ == "__main__":
    main()
