#!/usr/bin/env python
"""R4e-v2 — коммензурабельность укладки заряда 2π на дискретную цепь (архитектор).

Малые N (надёжная эквилибрация RT≈10): N−1∈{16,24,32,40,48,56,64} + пары чётности.
θ∈{0,90}, T_cold=3e-4, сектор Tw=2π (заряд фикс), пер-линк k_b=90,k_t=9 конст, 2 сида.
Пре-рег: (α) знак ΔS следует арифметике N−1; (β) стабилизация в классе / finite-size; (γ) ни то.
+ профиль ⟨τ̃_i⟩ по ветвям для двух N разной чётности (локализован/размазан заряд).
Результат — results/R4e_v2/report.md.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_pt_fns, classify, branch_counts
from ribbon_sim.frames import sector_sample, total_twist, quat_conj_mul
from ribbon_sim.pt import make_pt, geometric_ladder

# N−1 базовые (кратны 8) + пары чётности (добавляем N−1 нечётные у трёх точек)
NM1_BASE = [16, 24, 32, 40, 48, 56, 64]
NM1_PARITY = [17, 33, 49]            # нечётные пары к 16,32,48
NM1_GRID = sorted(set(NM1_BASE + NM1_PARITY))
N_GRID = [nm1 + 1 for nm1 in NM1_GRID]
THETA_DEG = [0.0, 90.0]
PROFILE_N = [33, 34]                 # два N разной чётности N−1 (32 чёт, 33 нечёт) для профиля τ̃
T_COLD = 3e-4
SWAP_EVERY = 10
B_RUN = 1024
N_BLOCKS = 6000
K_B, K_T = 90.0, 9.0
OUT = ROOT / "results" / "R4e_v2"; OUT.mkdir(parents=True, exist_ok=True)
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
    ef, sf = build_pt_fns(cfg, a, b, tw); run = make_pt(ef, sf, betas, SWAP_EVERY)
    q0, _ = sector_sample(jax.random.PRNGKey(0), (B, cfg["N"]), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B, len(betas), cfg["N"], 4))
    _, d = run(jax.random.PRNGKey(1), st0, nb)
    return np.asarray(d["accept_frac"])


def adaptive_ladder(cfg, tw, T_hot=1.5, target=0.35, max_R=40):
    Ts = list(np.geomspace(T_hot, T_COLD, 14)); acc = None
    for _ in range(30):
        betas = 1.0 / jnp.asarray(sorted(Ts, reverse=True))
        acc = pilot_acc(cfg, betas, tw)
        if acc.min() >= target or len(Ts) >= max_R:
            break
        j = int(np.argmin(acc)); Ts = sorted(Ts, reverse=True)
        Ts.insert(j + 1, np.sqrt(Ts[j] * Ts[j + 1]))
    return 1.0 / jnp.asarray(sorted(Ts, reverse=True)), len(sorted(Ts)), float(acc.min())


def run_point(cfg, key, theta, tw, betas, nb, want_profile=False):
    a, b = setting(np.radians(theta))
    ef, sf = build_pt_fns(cfg, a, b, tw); run = make_pt(ef, sf, betas, SWAP_EVERY)
    k1, k2 = jax.random.split(key)
    q0, twe = sector_sample(k1, (B_RUN, cfg["N"]), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B_RUN, len(betas), cfg["N"], 4))
    st, d = run(k2, st0, nb)
    qc = st[:, -1]
    s, t = classify(qc, a, b); s = np.asarray(s); t = np.asarray(t)
    cnt = np.asarray(branch_counts(jnp.asarray(s), jnp.asarray(t)))
    prof = None
    if want_profile:
        tau = 2.0 * np.asarray(quat_conj_mul(qc[:, :-1], qc[:, 1:])[..., 3])  # (B, N-1)
        stp = s * t
        prof = {"maj": tau[stp > 0].mean(0) if (stp > 0).sum() > 10 else None,
                "mino": tau[stp < 0].mean(0) if (stp < 0).sum() > 10 else None,
                "all": tau.mean(0)}
    return cnt, d, float(twe), prof


def P_anti(cnt):
    c = cnt.astype(float); return (c[1] + c[2]) / max(c.sum(), 1)


def dS_of(cnt):
    p = np.clip(P_anti(cnt), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def main():
    tw = 2 * np.pi
    rows.append("# R4e-v2 — коммензурабельность укладки заряда 2π (твист/линк=2π/(N−1))\n")
    rows.append(f"N−1={NM1_GRID}, θ={THETA_DEG}, T_cold={T_COLD}, сектор Tw=2π, пер-линк k_b={K_B} k_t={K_T} "
                f"конст, PT (свопы/{SWAP_EVERY}, {N_BLOCKS}бл, B={B_RUN}, 2 сида).\n")
    rows.append("Пре-рег: (α) знак ΔS следует арифметике N−1; (β) стабилизация в классе / finite-size; "
                "(γ) ни то — выделенный прогон.\n")
    t0 = time.time()
    res = {}; profiles = {}
    for N in N_GRID:
        cfg = cfg_for(N); betas, R, accmin = adaptive_ladder(cfg, tw)
        for th in THETA_DEG:
            for seed in [0, 1]:
                wp = (N in PROFILE_N and th == 0.0 and seed == 0)
                r = run_point(cfg, jax.random.PRNGKey(hash((N, th, seed)) % (2**31)), th, tw, betas, N_BLOCKS, wp)
                res[(N, th, seed)] = r
                if wp and r[3]:
                    profiles[N] = r[3]
        rt = float(res[(N, 0.0, 0)][1]["mean_roundtrips"])
        print(f"N={N} (N−1={N-1}): R={R} RT={rt:.1f} ΔS(0)={dS_of(sum(res[(N,0.0,s)][0] for s in[0,1])):+.3f} ({(time.time()-t0)/60:.1f}мин)")
        flush()

    # --- таблица ΔS(N−1) с чётностью и mod-классами ---
    rows.append("## ΔS(θ=0) по N−1: знак и арифметика\n")
    rows.append("| N−1 | N | RT | ΔS(0°) | знак | N−1 mod2 | mod3 | mod4 | сид-Δ |")
    rows.append("|---|---|---|---|---|---|---|---|---|")
    dS0 = {}
    for N in N_GRID:
        nm1 = N - 1
        c = sum(res[(N, 0.0, s)][0] for s in [0, 1]); dS0[nm1] = dS_of(c)
        sg = abs(dS_of(res[(N, 0.0, 0)][0]) - dS_of(res[(N, 0.0, 1)][0]))
        rt = float(res[(N, 0.0, 0)][1]["mean_roundtrips"])
        sgn = "+" if dS0[nm1] > 0.05 else ("−" if dS0[nm1] < -0.05 else "0")
        rows.append(f"| {nm1} | {N} | {rt:.1f} | {dS0[nm1]:+.3f} | {sgn} | {nm1%2} | {nm1%3} | {nm1%4} | {sg:.3f} |")
    flush()

    # --- анализ арифметики ---
    nm1s = sorted(dS0.keys()); vals = np.array([dS0[k] for k in nm1s])
    rows.append("\n## Анализ: следует ли знак арифметике N−1?\n")
    for mod, name in [(2, "чётность"), (3, "mod-3"), (4, "mod-4")]:
        # разброс знаков внутри классов
        classes = {}
        for k in nm1s:
            classes.setdefault(k % mod, []).append(dS0[k])
        consistent = all(len(set(np.sign([x for x in v if abs(x) > 0.05]))) <= 1 for v in classes.values() if any(abs(x) > 0.05 for x in v))
        cls_str = "; ".join(f"{c}:{np.round(v,2).tolist()}" for c, v in sorted(classes.items()))
        rows.append(f"- **{name} (mod {mod}):** классы [{cls_str}] — знак {'СОГЛАСОВАН в классах ✅' if consistent else 'НЕ согласован ❌'}.")
    flush()

    # --- профиль τ̃ по ветвям (локализация заряда) ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(profiles), figsize=(6*max(len(profiles),1), 4), squeeze=False)
    for i, (N, pr) in enumerate(sorted(profiles.items())):
        ax = axes[0][i]
        ax.axhline(2*np.pi/(N-1), color="gray", ls="--", lw=1, label="равномерно 2π/(N−1)")
        if pr["all"] is not None: ax.plot(pr["all"], "k-", lw=1, label="⟨τ̃_i⟩ все")
        if pr["maj"] is not None: ax.plot(pr["maj"], "C0-", label="мажор ветвь")
        if pr["mino"] is not None: ax.plot(pr["mino"], "C3-", label="минор ветвь")
        ax.set_title(f"N={N} (N−1={N-1}, {'чёт' if (N-1)%2==0 else 'нечёт'})")
        ax.set_xlabel("№ связи i"); ax.set_ylabel("⟨τ̃_i⟩"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Профиль скрутки по ветвям (локализация/размазка заряда 2π)")
    fig.tight_layout(); fig.savefig(OUT / "twist_profile.png", dpi=120); plt.close(fig)
    rows.append("\n## Профиль ⟨τ̃_i⟩ по ветвям (два N разной чётности)\n![профиль](twist_profile.png)\n")
    for N, pr in sorted(profiles.items()):
        loc = "РАЗМАЗАН (плоский ~2π/(N−1))" if pr["all"] is not None and np.std(pr["all"]) < 0.3*np.mean(np.abs(pr["all"])+1e-9) else "ЛОКАЛИЗОВАН (пики)"
        rows.append(f"- N={N} (N−1={N-1}): заряд {loc}.")

    # --- вердикт ---
    rows.append("\n## Вердикт по пре-регистрации\n")
    # проверим согласованность знака в классах чётности
    even = [dS0[k] for k in nm1s if k % 2 == 0 and abs(dS0[k]) > 0.05]
    odd = [dS0[k] for k in nm1s if k % 2 == 1 and abs(dS0[k]) > 0.05]
    par_consistent = (len(set(np.sign(even))) <= 1 if even else True) and (len(set(np.sign(odd))) <= 1 if odd else True)
    twe_max = max(res[(N, th, s)][2] for N in N_GRID for th in THETA_DEG for s in [0, 1])
    if par_consistent and (even or odd):
        rows.append("> **(α) ЗНАК СЛЕДУЕТ АРИФМЕТИКЕ N−1** (чётность-согласован) — коммензурабельный сценарий. "
                    "N≥96-нули = гашение укладок. Дискретная укладка 2π задаёт знак.")
    else:
        rows.append("> **(γ) знак НЕ следует простой арифметике** (чёт/mod непоследовательны) — нужен "
                    "выделенный длинный прогон N≥96 для finite-size vs недоперемешан.")
    rows.append(f"\nКонтроль Tw: max|ΔTw|={twe_max:.1e}. Время: {(time.time()-t0)/60:.1f} мин.")
    flush()
    print(f"R4e-v2 готово: ΔS(N−1)={[(k,round(dS0[k],2)) for k in nm1s]}")


if __name__ == "__main__":
    main()
