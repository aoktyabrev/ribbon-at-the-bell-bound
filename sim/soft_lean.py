#!/usr/bin/env python
"""Мягкая лента (lean-версия, закрытие полноты — последний непротестированный режим).

Одна ячейка k_e=1≈T0, spinor, отжиг decay=0.9995. Диагностики: h-популяции, E(θ|h), M2,
ВРЕМЕННЫЕ РЯДЫ M1/M2 по ходу отжига на θ∈{45,90}. Мягкая лента (k_e≈T0) — единственный режим,
где отжиг МОЖЕТ населить кинки (в отличие от жёсткой k_e=63, где энергия кинка ≫ T0).
Результат — results/soft/report.md.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_relaxer, classify, branch_counts, holonomy, kink_count
from ribbon_sim.frames import haar_quaternions
from ribbon_sim.experiment import setting_vectors

N, B = 64, 8192
K_E = 1.0          # мягкая: k_e ≈ T0
T0, DECAY = 1.0, 0.9995
BLOCK, CEILING = 25000, 200000
COLD_T = 1e-4
THETA_DEG = list(np.arange(0, 180.01, 15.0))       # 13 точек
TS_THETA = [45.0, 90.0]                            # временные ряды M1/M2
OUT = ROOT / "results" / "soft"; OUT.mkdir(parents=True, exist_ok=True)
rows = []


def cfg(steps):
    return {"N": N, "B": B, "k_e": K_E, "k_c": 1.0, "spinor": True, "elastic": "spinor",
            "T0": T0, "decay": DECAY, "steps": steps, "lr": 0.5 / (K_E + 1.0)}


def flush():
    (OUT / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def npbc(s, t):
    return np.array([np.sum((s > 0) & (t > 0)), np.sum((s > 0) & (t < 0)),
                     np.sum((s < 0) & (t > 0)), np.sum((s < 0) & (t < 0))])


def relax_theta(theta, seed, record_ts=False):
    """Блочный отжиг до T<COLD_T; опц. временной ряд (E_M1,E_M2,плотн.кинков) по блокам."""
    a, b = setting_vectors(np.radians(theta))
    run = build_relaxer(cfg(BLOCK))["run"]
    key = jax.random.fold_in(jax.random.PRNGKey(seed), int(theta * 10))
    ki, kn = jax.random.split(key)
    q = haar_quaternions(ki, (B, N))
    total = 0; ts = []
    while total < CEILING:
        kn, sub = jax.random.split(kn)
        q, _ = run(sub, q, a, b, jnp.int32(total))
        total += BLOCK
        if record_ts:
            s, t = classify(q, a, b); s = np.asarray(s); t = np.asarray(t)
            h = np.asarray(holonomy(q))
            e_m1 = float(np.mean(s * t)); e_m2 = float(np.mean(s * (h * t)))
            kd = float(np.mean(kink_count(q))) / (N - 1)
            ts.append((total, e_m1, e_m2, kd))
        T = T0 * DECAY ** total
        if T < COLD_T:
            break
    s, t = classify(q, a, b); s = np.asarray(s); t = np.asarray(t)
    h = np.asarray(holonomy(q))
    kd = float(np.mean(kink_count(q))) / (N - 1)
    cnt = npbc(s, t); cnt_m2 = npbc(s, h * t)
    n_hm = int(np.sum(h < 0))
    c_hp = npbc(s[h > 0], t[h > 0]); c_hm = npbc(s[h < 0], t[h < 0])
    return {"cnt": cnt, "cnt_m2": cnt_m2, "kd": kd, "n_hm": n_hm, "P_hm": n_hm / (B),
            "c_hp": c_hp, "c_hm": c_hm, "ts": ts, "steps": total}


def E(cnt):
    c = cnt.astype(float); n = c.sum()
    return (c[0] + c[3] - c[1] - c[2]) / max(n, 1)


def main():
    rows.append("# Мягкая лента (lean): k_e=1≈T0, спинор, отжиг — населяет ли кинки\n")
    rows.append(f"k_e={K_E} (мягкая), N={N}, B={B}, отжиг T0={T0} decay={DECAY} до T<{COLD_T}. "
                "Единственный режим, где энергия кинка ~k_e·arccos²~2-10 сопоставима с T0=1 ⇒ отжиг "
                "МОЖЕТ населить кинки (жёсткая k_e=63 не могла, R5b).\n")
    t0 = time.time()

    # свип θ, 2 сида
    res = {}
    for seed in [0, 1]:
        for th in THETA_DEG:
            res[(seed, th)] = relax_theta(th, seed, record_ts=(seed == 0 and th in TS_THETA))
        print(f"  сид{seed} готов ({(time.time()-t0)/60:.1f}мин)")
        flush()

    # --- сводка M1/M2/h ---
    rows.append("## E(θ), доля антиферро, голономия\n")
    rows.append("| θ° | E_M1 | E_M2 | плотн.кинков | P(h=−1) |")
    rows.append("|---|---|---|---|---|")
    for th in THETA_DEG:
        r = res[(0, th)]
        rows.append(f"| {th:g} | {E(r['cnt']):+.3f} | {E(r['cnt_m2']):+.3f} | {r['kd']:.4f} | {r['P_hm']:.4f} |")
    flush()

    # --- ключевой вопрос: населены ли кинки? ---
    kd_max = max(res[(0, th)]["kd"] for th in THETA_DEG)
    phm_max = max(res[(0, th)]["P_hm"] for th in THETA_DEG)
    rows.append(f"\n## Населены ли кинки (h=−1)?\n")
    rows.append(f"max плотность кинков={kd_max:.4f}, max P(h=−1)={phm_max:.4f}. "
                + ("✅ КИНКИ НАСЕЛЕНЫ (мягкая лента + отжиг): голономия h=−1 существует ⇒ M2≠M1 возможно."
                   if phm_max > 0.01 else
                   "❌ кинки НЕ населены даже в мягкой ленте (энергия кинка всё ещё > T0 при сходимости)."))
    flush()

    # --- E(θ|h) если есть h=−1 ---
    if phm_max > 0.01:
        rows.append("\n## Условная E(θ|h) (где P(h=−1)>0)\n")
        rows.append("| θ° | E(θ|h=+1) | E(θ|h=−1) |")
        rows.append("|---|---|---|")
        for th in THETA_DEG:
            r = res[(0, th)]
            ehp = E(r["c_hp"]) if r["c_hp"].sum() > 10 else float("nan")
            ehm = E(r["c_hm"]) if r["c_hm"].sum() > 10 else float("nan")
            if r["c_hm"].sum() > 10:
                rows.append(f"| {th:g} | {ehp:+.3f} | {ehm:+.3f} |")
        flush()

    # --- временные ряды M1/M2 ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(TS_THETA), figsize=(6 * len(TS_THETA), 4), squeeze=False)
    for i, th in enumerate(TS_THETA):
        ts = np.array(res[(0, th)]["ts"])
        ax = axes[0][i]
        if len(ts):
            ax.plot(ts[:, 0], ts[:, 1], "o-", label="E_M1 (оси)")
            ax.plot(ts[:, 0], ts[:, 2], "s-", label="E_M2 (голоном.)")
            ax.plot(ts[:, 0], ts[:, 3], "^-", label="плотн.кинков", alpha=0.6)
        ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("шаг отжига"); ax.set_title(f"θ={th:g}°")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Мягкая лента: временные ряды M1/M2 по ходу отжига")
    fig.tight_layout(); fig.savefig(OUT / "timeseries.png", dpi=120); plt.close(fig)
    rows.append("\n## Временные ряды M1/M2 (θ∈{45,90})\n![временные ряды](timeseries.png)\n")

    # --- воспроизводимость по 2 сидам ---
    dE = max(abs(E(res[(0, th)]["cnt"]) - E(res[(1, th)]["cnt"])) for th in THETA_DEG)
    rows.append(f"## Контроль\nВоспроизводимость 2 сидов: max|ΔE_M1|={dE:.3f}. "
                f"Знак M1@0: E={E(res[(0,0.0)]['cnt']):+.3f} ({'ферро' if E(res[(0,0.0)]['cnt'])>0 else 'антиферро'}).")
    rows.append(f"\n---\nВремя: {(time.time()-t0)/60:.1f} мин.")
    flush()
    print(f"soft-lean готово: kink_dens_max={kd_max:.4f} P(h=-1)_max={phm_max:.4f} E_M1(0)={E(res[(0,0.0)]['cnt']):+.3f}")


if __name__ == "__main__":
    main()
