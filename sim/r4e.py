#!/usr/bin/env python
"""R4e — экстенсивность энтропийного зазора ΔS по N (решающее для леммы, архитектор).

N∈{32,64,128}, θ∈{0,60,90}, T_cold=3e-4, сектор Tw=2π (заряд тот же 2π, не масштабируется),
k_t/k_b=0.1, PT-протокол R4c. Извлечь ΔS(θ,N)=ln[P/(1−P)] (при ΔE≈0: ΔF=−T·ΔS).
Пре-рег: (а) ΔS∝N ⇒ ножницы по N; (б) ΔS≈const(N) ⇒ амплитуда топологически заперта на tanh(ΔS/2).
Результат — results/R4e/report.md.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_pt_fns, classify, branch_counts
from ribbon_sim.frames import sector_sample, total_twist
from ribbon_sim.pt import make_pt, geometric_ladder

N_GRID = [32, 64, 96, 128]
THETA_DEG = [0.0, 60.0, 90.0]
T_COLD = 3e-4
SWAP_EVERY = 10
B_RUN = 1024
N_BLOCKS = 6000
# ПЕР-ЛИНК жёсткости КОНСТАНТНЫ (материал один, меняется длина) — фикс архитектора.
# Значения из N=64 κ=1 (k_b=90, k_t=9); при других N κ меняется, что и нужно для теста экстенсивности.
K_B, K_T = 90.0, 9.0
OUT = ROOT / "results" / "R4e"; OUT.mkdir(parents=True, exist_ok=True)
rows = []


def cfg_for(N):
    return {"lr": 0.5 / (max(K_B, K_T) + 1.0), "k_e": 0.0, "k_c": 1.0, "spinor": True,
            "elastic": "cosserat_geo", "k_b": K_B, "k_t": K_T, "twist_project": True,
            "n_twist_corr": 10, "N": N}


def flush():
    (OUT / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def setting(theta):
    return (jnp.array([0., 0., 1.]), jnp.array([np.sin(theta), 0., np.cos(theta)]))


def pilot_acc(cfg, betas, tw, B=256, nb=150):
    a, b = setting(np.radians(90.0))
    ef, sf = build_pt_fns(cfg, a, b, tw)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    q0, _ = sector_sample(jax.random.PRNGKey(0), (B, cfg["N"]), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B, len(betas), cfg["N"], 4))
    _, d = run(jax.random.PRNGKey(1), st0, nb)
    return np.asarray(d["accept_frac"])


def adaptive_ladder(cfg, tw, T_hot=1.5, target=0.35, max_R=56):
    """Адаптивная лестница равного acceptance (бисекция низких звеньев). T_hot=1.5 мостит
    ветвевую фрустрацию (энергошкала ~k_c=1); адаптивный шаг добавляет ступени под полную
    энергию ∝N (фикс архитектора реализован через рост числа ступеней, а не T_hot: поднятие
    T_hot при фикс. ступенях лишь разрежает лестницу и УХУДШАЕТ acceptance)."""
    Ts = list(np.geomspace(T_hot, T_COLD, 14))
    acc = None
    for _ in range(48):
        betas = 1.0 / jnp.asarray(sorted(Ts, reverse=True))
        acc = pilot_acc(cfg, betas, tw)
        if acc.min() >= target or len(Ts) >= max_R:
            break
        j = int(np.argmin(acc)); Ts = sorted(Ts, reverse=True)
        Ts.insert(j + 1, np.sqrt(Ts[j] * Ts[j + 1]))
    Ts = sorted(Ts, reverse=True)
    return 1.0 / jnp.asarray(Ts), len(Ts), float(acc.min()), T_hot


def run_point(cfg, key, theta, tw, betas, nb):
    a, b = setting(np.radians(theta))
    ef, sf = build_pt_fns(cfg, a, b, tw)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    k1, k2 = jax.random.split(key)
    q0, twe = sector_sample(k1, (B_RUN, cfg["N"]), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B_RUN, len(betas), cfg["N"], 4))
    st, d = run(k2, st0, nb)
    s, t = classify(st[:, -1], a, b)
    return np.asarray(branch_counts(s, t)), d, float(twe)


def P_anti(cnt):
    c = cnt.astype(float); return (c[1] + c[2]) / max(c.sum(), 1)


def main():
    tw = 2 * np.pi
    rows.append("# R4e — экстенсивность энтропийного зазора ΔS по N\n")
    rows.append(f"Сектор Tw=2π (заряд ТОТ ЖЕ, не масштаб.), k_t/k_b=0.1, T_cold={T_COLD}, θ={THETA_DEG}, "
                f"N={N_GRID}, PT (свопы/{SWAP_EVERY}, {N_BLOCKS} бл, B={B_RUN}, 2 сида).\n")
    rows.append("ΔS(θ,N)=ln[P/(1−P)] (при ΔE≈0: ΔF=−T·ΔS). Пре-рег: (а) ΔS∝N (ножницы по N); "
                "(б) ΔS≈const(N) (амплитуда топологически заперта). Обе легальны.\n")
    t0 = time.time()
    res = {}          # (N, θ, seed) -> (cnt, diag, twe)
    lad_info = {}
    for N in N_GRID:
        cfg = cfg_for(N)
        betas, R, accmin, T_hot = adaptive_ladder(cfg, tw)
        lad_info[N] = (R, accmin, T_hot)
        print(f"N={N}: T_hot={T_hot} лестница R={R} min-acc={accmin:.2f} ({(time.time()-t0)/60:.1f}мин)")
        for th in THETA_DEG:
            for seed in [0, 1]:
                k = jax.random.PRNGKey(hash((N, round(th, 1), seed)) % (2**31))
                res[(N, th, seed)] = run_point(cfg, k, th, tw, betas, N_BLOCKS)
        rt = np.mean([float(res[(N, th, 0)][1]["mean_roundtrips"]) for th in THETA_DEG])
        print(f"  N={N}: T_hot={T_hot} mean-RT={rt:.1f} готово")
        flush()

    rows.append("## Калибровка лестниц по N (фикс: пер-линк k_b=90,k_t=9 конст., T_hot по N)\n")
    rows.append("| N | T_hot | R | min-acc | mean-RT | сид-согласие ΔS(0) |")
    rows.append("|---|---|---|---|---|---|")
    for N in N_GRID:
        R, accmin, T_hot = lad_info[N]
        rt = np.mean([float(res[(N, th, 0)][1]["mean_roundtrips"]) for th in THETA_DEG])
        # сид-согласие в ΔS при θ=0
        dsA = [np.log(np.clip(P_anti(res[(N, 0.0, s)][0]), 1e-4, 1-1e-4) /
                      (1-np.clip(P_anti(res[(N, 0.0, s)][0]), 1e-4, 1-1e-4))) for s in [0, 1]]
        rows.append(f"| {N} | {T_hot:g} | {R} | {accmin:.2f} | {rt:.1f} | Δ={abs(dsA[0]-dsA[1]):.3f} |")
    flush()

    # --- ΔS(θ,N) ---
    rows.append("## ΔS(θ,N) = ln[P/(1−P)] по θ и N (2 сида усреднены)\n")
    rows.append("| N | mean-RT | ΔS(0°) | ΔS(60°) | ΔS(90°) | ΔS(0°)/cos0 |")
    rows.append("|---|---|---|---|---|---|")
    dS = {}
    for N in N_GRID:
        rt = np.mean([float(res[(N, th, 0)][1]["mean_roundtrips"]) for th in THETA_DEG])
        row = [f"{N}", f"{rt:.1f}"]
        for th in THETA_DEG:
            P = np.mean([P_anti(res[(N, th, s)][0]) for s in [0, 1]])
            P = np.clip(P, 1e-4, 1 - 1e-4)
            dS[(N, th)] = np.log(P / (1 - P))
            row.append(f"{dS[(N, th)]:+.3f}")
        row.append(f"{dS[(N, 0.0)]:+.3f}")  # cos0=1
        rows.append("| " + " | ".join(row) + " |")
    flush()

    # --- экстенсивность: фит const vs линейный по N (AIC + bootstrap CI наклона), θ=0 ---
    Ns = np.array(N_GRID, float)
    # усреднённые счётчики по сидам для θ=0, для bootstrap
    cnt0 = {N: sum(res[(N, 0.0, s)][0] for s in [0, 1]) for N in N_GRID}
    dS0 = np.array([np.log(np.clip(P_anti(cnt0[N]), 1e-4, 1-1e-4) /
                          (1-np.clip(P_anti(cnt0[N]), 1e-4, 1-1e-4))) for N in N_GRID])
    sigS = np.array([1.0 / np.sqrt(cnt0[N].sum() * np.clip(P_anti(cnt0[N]), 1e-3, 1-1e-3) *
                                   (1-np.clip(P_anti(cnt0[N]), 1e-3, 1-1e-3))) for N in N_GRID])
    w = 1.0 / sigS**2

    def fit_const():
        m = np.sum(w*dS0)/np.sum(w); return m, np.sum(w*(dS0-m)**2)

    def fit_lin():
        Amat = np.vstack([np.ones_like(Ns), Ns]).T
        WA = Amat*w[:, None]; coef = np.linalg.solve(Amat.T@WA, Amat.T@(w*dS0))
        chi2 = np.sum(w*(dS0-Amat@coef)**2); return coef, chi2
    mc, chi_c = fit_const(); (a_l, b_l), chi_l = fit_lin()
    aic_c = chi_c + 2*1; aic_l = chi_l + 2*2
    # bootstrap CI наклона: ресемпл биномиалов
    rng = np.random.default_rng(0); slopes = []
    for _ in range(2000):
        dSb = []
        for N in N_GRID:
            n = int(cnt0[N].sum()); p = np.clip(P_anti(cnt0[N]), 1e-4, 1-1e-4)
            pb = rng.binomial(n, p)/n; pb = np.clip(pb, 1e-4, 1-1e-4)
            dSb.append(np.log(pb/(1-pb)))
        dSb = np.array(dSb); Amat = np.vstack([np.ones_like(Ns), Ns]).T
        WA = Amat*w[:, None]; cb = np.linalg.solve(Amat.T@WA, Amat.T@(w*dSb)); slopes.append(cb[1])
    slopes = np.array(slopes); ci = (np.percentile(slopes, 2.5), np.percentile(slopes, 97.5))

    rows.append(f"\n## Вердикт: экстенсивность ΔS (const vs линейный, θ=0)\n")
    rows.append(f"ΔS(0°) по N={N_GRID}: {[f'{x:+.3f}±{s:.3f}' for x, s in zip(dS0, sigS)]}.")
    rows.append(f"- **const:** ΔS={mc:+.3f}, χ²={chi_c:.1f}, AIC={aic_c:.1f}.")
    rows.append(f"- **линейный:** ΔS={a_l:+.3f}{b_l:+.4f}·N, χ²={chi_l:.1f}, AIC={aic_l:.1f}.")
    rows.append(f"- наклон b = {b_l:+.4f}, bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}].")
    slope_zero = ci[0] <= 0 <= ci[1]
    better = "const" if aic_c < aic_l else "линейный"
    rows.append(f"- Лучшая по AIC: **{better}** (ΔAIC={abs(aic_c-aic_l):.1f}). Наклон {'совместим с 0' if slope_zero else 'ОТЛИЧЕН от 0'} (CI).")
    if better == "const" or slope_zero:
        A_lock = np.tanh(abs(mc)/2)
        rows.append(f"\n> **(б) ΔS ≈ const(N) — ИНТЕНСИВНА.** Зазор не зависит от длины (наклон CI включает 0, "
                    f"AIC-лучшая const). Амплитуда ТОПОЛОГИЧЕСКИ ЗАПЕРТА на A=tanh(ΔS/2)≈{A_lock:.3f}. "
                    "Лемма §3 в форме (i): топологический заряд задаёт зазор независимо от N. "
                    "Контроль N=256 не требуется.")
        need_ctrl = False
    else:
        rows.append(f"\n> **(а) ΔS ∝ N — ЭКСТЕНСИВНА** (наклон {b_l:+.4f}, CI не включает 0, AIC-лучшая линейный). "
                    "Амплитуда управляема размером, «ножницы по N». ТРЕБУЕТСЯ контроль N=256, θ=60 "
                    "(форма должна уходить от cos).")
        need_ctrl = True

    twe_max = max(res[(N, th, s)][2] for N in N_GRID for th in THETA_DEG for s in [0, 1])
    rows.append(f"\nКонтроль Tw: max|ΔTw|={twe_max:.1e} (сектор сохранён).")

    # --- пре-регистрированный контроль N=256 θ=60 (только если экстенсивна) ---
    if need_ctrl:
        cfg = cfg_for(256); betas, R, accmin, T_hot = adaptive_ladder(cfg, tw)
        c256 = sum(run_point(cfg, jax.random.PRNGKey(256*10+s), 60.0, tw, betas, N_BLOCKS)[0] for s in [0, 1])
        # форма: E при θ=60 vs синглет-ожидание (масштаб). Уход от cos = |E(60)/E-ожид| != cos(60)
        e256 = E(c256)
        rows.append(f"\n## Контроль N=256, θ=60° (форма при экстенсивной ΔS)\n")
        rows.append(f"E(60°,N=256)={e256:+.3f}. Форма {'уходит от cos (ножницы по N подтверждены)' if abs(e256)>0.6 else 'ещё косинусная'}.")
        flush()

    rows.append(f"\n---\nВремя: {(time.time()-t0)/60:.1f} мин.")
    flush()
    print(f"R4e готово: ΔS(0) по N={np.round(dS0,3)}, наклон={b_l:+.4f} CI[{ci[0]:+.3f},{ci[1]:+.3f}], лучшая={better}")


def E(cnt):
    c = cnt.astype(float); n = c.sum(); return (c[0]+c[3]-c[1]-c[2])/max(n, 1)


if __name__ == "__main__":
    main()
