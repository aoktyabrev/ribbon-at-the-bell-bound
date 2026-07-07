#!/usr/bin/env python
"""АУДИТ немонотонности R4a (архитектор, до R4b). Изгиб-жёсткая ячейка k_t/k_b=0.1,
сектор Tw=0, cosserat_geo+spinor, связь Tw=const.
1) Точечная сходимость θ∈{0,15,30}, потолок 600k, тренд E по блокам:
   E(0)→~1 ⇒ транзиент; стабилен ⇒ реальный constrained trapping.
2) Энергоаудит по ветвям на финалах θ=0: остаточная E (изгиб/твист) минор vs мажор;
   профиль n_i·a вдоль цепочки (гипотеза π-дуга) и τ̃_i (компенсирующая скрутка).
Результат — results/R4a/audit.md. Инкрементальная запись.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_relaxer, classify
from ribbon_sim.frames import (
    axis, geodesic, haar_quaternions, mirror_flip, quat_conj_mul,
)
from ribbon_sim.experiment import setting_vectors

N, B = 64, 8192
kappa_eq = 1.0; k_e_eq = kappa_eq * (N - 1) * 1.0
total = 3.0 * k_e_eq; r = 0.1
k_b = total / (2.0 + r); k_t = r * k_b        # изгиб-жёсткая
lr = 0.5 / (max(k_b, k_t) + 1.0)
cell = {"N": N, "B": B, "k_e": 0.0, "k_c": 1.0, "spinor": True, "elastic": "cosserat_geo",
        "k_b": k_b, "k_t": k_t, "T0": 0.0, "decay": 1.0, "steps": 25000, "lr": lr,
        "twist_project": True}
OUT = ROOT / "results" / "R4a" / "audit.md"; rows = []


def flush():
    OUT.write_text("\n".join(rows) + "\n", encoding="utf-8")


def mirror_init(key):
    q = haar_quaternions(key, (B, N))
    return q.at[B // 2:].set(mirror_flip(q[:B // 2]))


def relax_blocks(q, a, b, ceiling, block=25000):
    """Релаксация блоками до потолка; возвращает список (шаг, E) и финал q."""
    run = build_relaxer({**cell, "steps": block})["run"]
    key = jax.random.PRNGKey(123)
    trace = []
    tot = 0
    while tot < ceiling:
        key, sub = jax.random.split(key)
        q, _ = run(sub, q, a, b, jnp.int32(tot))
        tot += block
        s, t = classify(q, a, b)
        trace.append((tot, float(np.mean(np.asarray(s) * np.asarray(t)))))
    return trace, q


def main():
    t0 = time.time()
    rows.append("# R4a — аудит немонотонности (изгиб-жёсткая k_t/k_b=0.1, Tw=0)\n")
    rows.append(f"cosserat_geo+spinor, N={N}, B={B}, связь Tw=0, зеркальные пары. "
                f"k_b={k_b:.1f}, k_t={k_t:.2f}.\n")

    # ---- 1. точечная сходимость до 600k ----
    rows.append("## 1. Точечная сходимость E(θ) до 600k шагов\n")
    rows.append("| θ° | E@100k | E@200k | E@400k | E@600k | Δ(600k−100k) |")
    rows.append("|---|---|---|---|---|---|")
    finals = {}
    e0_100 = e0_600 = np.nan
    e30_100 = e30_600 = np.nan
    for thd in (0.0, 15.0, 30.0):
        a, b = setting_vectors(np.radians(thd))
        trace, qf = relax_blocks(mirror_init(jax.random.PRNGKey(1)), a, b, 600000)
        finals[thd] = (qf, a, b)
        Ev = dict(trace)
        g = lambda k: Ev.get(k, np.nan)  # noqa: E731
        if thd == 0.0:
            e0_100, e0_600 = g(100000), g(600000)
        if thd == 30.0:
            e30_100, e30_600 = g(100000), g(600000)
        rows.append(f"| {thd:.0f} | {g(100000):+.3f} | {g(200000):+.3f} | {g(400000):+.3f} "
                    f"| {g(600000):+.3f} | {g(600000)-g(100000):+.3f} |")
        flush(); print(f"θ={thd}: E 100k→600k = {g(100000):+.3f}→{g(600000):+.3f}")

    # немонотонность = E(30) − E(0); транзиент если E(0)→1 и |немонотонность| исчезает
    nonmono_100 = e30_100 - e0_100
    nonmono_600 = e30_600 - e0_600
    if e0_600 - e0_100 > 0.03 and abs(nonmono_600) < abs(nonmono_100) - 0.02:
        verdict1 = (f"ТРАНЗИЕНТ: E(0) ползёт {e0_100:+.3f}→{e0_600:+.3f}, немонотонность "
                    f"(E30−E0) спадает {nonmono_100:+.3f}→{nonmono_600:+.3f} — хоронится")
    elif abs(e0_600 - e0_100) < 0.02:
        verdict1 = (f"СТАБИЛЕН: E(0)={e0_600:+.3f} (Δ={e0_600-e0_100:+.3f}), немонотонность "
                    f"E30−E0={nonmono_600:+.3f} держится — РЕАЛЬНЫЙ constrained trapping")
    else:
        verdict1 = (f"ЧАСТИЧНЫЙ дрейф: E(0) Δ={e0_600-e0_100:+.3f}, немонотонность "
                    f"{nonmono_100:+.3f}→{nonmono_600:+.3f} — к архитектору")
    rows.append(f"\n**Вердикт (1):** {verdict1}.\n")
    flush()

    # ---- 2. энергоаудит по ветвям на финале θ=0 ----
    rows.append("## 2. Энергоаудит ветвей на финале θ=0 (600k)\n")
    qf, a, b = finals[0.0]
    s, t = classify(qf, a, b); s = np.asarray(s); t = np.asarray(t)
    stp = (s * t)  # +1 same (мажор ожидается), -1 opp
    # изгиб/твист на ленту
    rlink = np.asarray(quat_conj_mul(qf[:, :-1], qf[:, 1:]))
    d = np.asarray(geodesic(qf[:, :-1], qf[:, 1:], spinor=True))
    x, y, z = rlink[..., 1], rlink[..., 2], rlink[..., 3]
    tfrac = z * z / (x * x + y * y + z * z + 1e-12)
    # E_i = k_b·d² + (k_t−k_b)·d²·tfrac ⇒ изгиб-часть k_b·d², твист-часть (k_t−k_b)·d²·tfrac
    e_bend = np.sum(k_b * d * d, axis=1)
    e_tw = np.sum((k_t - k_b) * d * d * tfrac, axis=1)
    maj = stp > 0; mino = stp < 0
    rows.append(f"Доля мажор (s·t=+1): {maj.mean():.3f}; минор: {mino.mean():.3f}.\n")
    rows.append("| ветвь | ⟨E_изгиб⟩ | ⟨E_твист⟩ | ⟨E_полн⟩ |")
    rows.append("|---|---|---|---|")
    for lbl, mask in (("мажор (s·t=+1)", maj), ("минор (s·t=−1)", mino)):
        if mask.sum() > 10:
            rows.append(f"| {lbl} | {e_bend[mask].mean():.2f} | {e_tw[mask].mean():.2f} "
                        f"| {(e_bend+e_tw)[mask].mean():.2f} |")
    flush()

    # профиль n_i·a вдоль цепочки (минор vs мажор) + τ̃_i
    nia = np.asarray(jax.vmap(lambda Q: jax.vmap(lambda q1: jnp.dot(axis(q1), a))(Q))(qf))  # (B,N)
    tau = 2.0 * z  # (B, N-1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if maj.sum() > 10:
        ax1.plot(nia[maj].mean(0), "C0-", label="мажор ⟨n_i·a⟩")
        ax2.plot(tau[maj].mean(0), "C0-", label="мажор ⟨τ̃_i⟩")
    if mino.sum() > 10:
        ax1.plot(nia[mino].mean(0), "C3-", label="минор ⟨n_i·a⟩ (π-дуга?)")
        ax2.plot(tau[mino].mean(0), "C3-", label="минор ⟨τ̃_i⟩ (компенс.?)")
    ax1.axhline(0, color="gray", lw=0.5); ax1.set_xlabel("№ фрейма i"); ax1.set_ylabel("n_i·a")
    ax1.set_title("Профиль оси вдоль ленты (θ=0)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.axhline(0, color="gray", lw=0.5); ax2.set_xlabel("№ связи i"); ax2.set_ylabel("τ̃_i")
    ax2.set_title("Профиль скрутки (θ=0)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "R4a" / "audit_profiles.png", dpi=120); plt.close(fig)
    rows.append("\n![профили минор/мажор](audit_profiles.png)\n")
    rows.append("Гипотезы: минор-ленты несут π-дугу в n_i·a (концы антивыровнены внутри) и "
                "компенсирующую скрутку τ̃_i при Tw=0.\n")
    rows.append(f"\n---\nВремя аудита: {time.time()-t0:.0f} с.")
    flush(); print(f"audit → {OUT}")


if __name__ == "__main__":
    main()
