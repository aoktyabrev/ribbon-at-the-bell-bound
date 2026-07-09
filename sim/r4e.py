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

N_GRID = [32, 64, 128]
THETA_DEG = [0.0, 60.0, 90.0]
T_COLD = 3e-4
SWAP_EVERY = 10
B_RUN = 1024
N_BLOCKS = 6000
OUT = ROOT / "results" / "R4e"; OUT.mkdir(parents=True, exist_ok=True)
rows = []


def cfg_for(N):
    total = 3.0 * (N - 1)           # cosserat_geo: 2k_b+k_t=3·k_e_eq, κ=1
    k_b = total / 2.1; k_t = 0.1 * k_b
    return {"lr": 0.5 / (max(k_b, k_t) + 1.0), "k_e": 0.0, "k_c": 1.0, "spinor": True,
            "elastic": "cosserat_geo", "k_b": k_b, "k_t": k_t, "twist_project": True,
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


def adaptive_ladder(cfg, tw, T_hot=1.5, target=0.35, max_R=44):
    Ts = list(np.geomspace(T_hot, T_COLD, 12))
    acc = None
    for _ in range(30):
        betas = 1.0 / jnp.asarray(sorted(Ts, reverse=True))
        acc = pilot_acc(cfg, betas, tw)
        if acc.min() >= target or len(Ts) >= max_R:
            break
        j = int(np.argmin(acc)); Ts = sorted(Ts, reverse=True)
        Ts.insert(j + 1, np.sqrt(Ts[j] * Ts[j + 1]))
    Ts = sorted(Ts, reverse=True)
    return 1.0 / jnp.asarray(Ts), len(Ts), float(acc.min())


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
    for N in N_GRID:
        cfg = cfg_for(N)
        betas, R, accmin = adaptive_ladder(cfg, tw)
        print(f"N={N}: лестница R={R} min-acc={accmin:.2f} ({(time.time()-t0)/60:.1f}мин)")
        for th in THETA_DEG:
            for seed in [0, 1]:
                k = jax.random.PRNGKey(hash((N, round(th, 1), seed)) % (2**31))
                res[(N, th, seed)] = run_point(cfg, k, th, tw, betas, N_BLOCKS)
        rt = np.mean([float(res[(N, th, 0)][1]["mean_roundtrips"]) for th in THETA_DEG])
        print(f"  N={N}: mean-RT={rt:.1f} готово")
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

    # --- экстенсивность: ΔS(0°) vs N ---
    Ns = np.array(N_GRID, float); dS0 = np.array([dS[(N, 0.0)] for N in N_GRID])
    # фит ΔS0 = a + b·N; и ΔS0 = const
    A = np.vstack([np.ones_like(Ns), Ns]).T; coef, *_ = np.linalg.lstsq(A, dS0, rcond=None)
    slope_per_N = coef[1]
    rel_spread = (dS0.max() - dS0.min()) / abs(dS0.mean())
    rows.append(f"\n## Вердикт: экстенсивность ΔS\n")
    rows.append(f"ΔS(0°) по N={N_GRID}: {[f'{x:+.3f}' for x in dS0]}.")
    rows.append(f"- Линейный фит ΔS0 = {coef[0]:+.3f} + {slope_per_N:+.4f}·N; относит. разброс {rel_spread:.2%}.")
    extensive = abs(slope_per_N) * (Ns.max() - Ns.min()) > 0.3 and rel_spread > 0.25
    if extensive:
        rows.append(f"\n> **(а) ΔS ∝ N (ЭКСТЕНСИВНА):** зазор растёт с N ⇒ амплитуда управляема размером, "
                    "но «ножницы по N» — форма уйдёт от cos при больших N (проверяемо следом). "
                    "Мера НЕ топологически заперта.")
    else:
        rows.append(f"\n> **(б) ΔS ≈ const(N) (ИНТЕНСИВНА):** зазор не зависит от N (разброс {rel_spread:.1%}) ⇒ "
                    f"амплитуда ТОПОЛОГИЧЕСКИ ЗАПЕРТА на A=tanh(ΔS/2)≈{np.tanh(abs(dS0.mean())/2):.3f}. "
                    "Топологический заряд задаёт зазор независимо от длины ленты — сильный результат для леммы §3.")
    twe_max = max(res[(N, th, s)][2] for N in N_GRID for th in THETA_DEG for s in [0, 1])
    rows.append(f"\nКонтроль Tw: max|ΔTw|={twe_max:.1e}. Время: {(time.time()-t0)/60:.1f} мин.")
    flush()
    print(f"R4e готово: ΔS(0) по N={np.round(dS0,3)}, slope/N={slope_per_N:.4f}, extensive={extensive}")


if __name__ == "__main__":
    main()
