"""Визуализация (SPEC §5): кривые E(θ), модели-соперники, гистограммы ветвей."""

import matplotlib

matplotlib.use("Agg")  # без дисплея (WSL/headless)
import matplotlib.pyplot as plt
import numpy as np

from . import analysis


def plot_E_curve(thetas, E, out_path, seeds_E=None, title="E(θ)"):
    """Кривая E(θ) с наложенными моделями-соперниками.

    thetas — в радианах; seeds_E — опц. список кривых E на отдельных сидах.
    """
    deg = np.degrees(thetas)
    grid = np.linspace(0, np.pi, 200)
    grid_deg = np.degrees(grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    # референсные модели
    ax.plot(grid_deg, -np.cos(grid), "k--", lw=1, alpha=0.6, label="−cos θ (КМ, p=2)")
    ax.plot(grid_deg, 2 * grid / np.pi - 1, "b:", lw=1, alpha=0.6, label="пила 2θ/π−1")
    Ep1 = analysis.E_from_counts(analysis.chord_probs(grid, 1.0) * 1e6)
    ax.plot(grid_deg, Ep1, "g-.", lw=1, alpha=0.5, label="хорда p=1")
    ax.axhline(0, color="gray", lw=0.5)

    if seeds_E is not None:
        for i, Es in enumerate(seeds_E):
            ax.plot(deg, Es, ".", ms=4, alpha=0.4, label=f"сид {i}")
    ax.plot(deg, E, "ro-", ms=5, lw=1.5, label="эмпирика E(θ)")

    ax.set_xlabel("θ, градусы")
    ax.set_ylabel("E(θ) = ⟨s·t⟩")
    ax.set_title(title)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_measure_diag(measure, out_path, title="диагностика меры"):
    """Гистограммы c_A=n_A·a, c_B=n_B·b на финалах: 40 бинов [−1,1], плотность (uniform-
    сфера = 0.5), раздельно по ветвям (s,t), по θ; оверлеи uniform / KS∝max(c,0) / δ-пик |c|=1 (R4)."""
    thetas = sorted(measure.keys())
    nT = len(thetas)
    fig, axes = plt.subplots(nT, 2, figsize=(11, 3.1 * nT), squeeze=False)
    cgrid = np.linspace(-1, 1, 200)
    branches = [(1, 1, "pp", "C0"), (1, -1, "pm", "C1"), (-1, 1, "mp", "C2"), (-1, -1, "mm", "C3")]
    for i, th in enumerate(thetas):
        m = measure[th]
        s, t = m["s"], m["t"]
        for j, (cname, cvals) in enumerate([("c_A=n_A·a", m["c_A"]), ("c_B=n_B·b", m["c_B"])]):
            ax = axes[i][j]
            for bs, bt, lbl, col in branches:
                mask = (s == bs) & (t == bt)
                if mask.sum() > 20:
                    ax.hist(np.asarray(cvals)[mask], bins=40, range=(-1, 1), density=True,
                            histtype="step", color=col, alpha=0.75, label=lbl)
            ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6, label="uniform (0.5)")
            ax.plot(cgrid, 2 * np.maximum(cgrid, 0), "m:", lw=1.5, alpha=0.7, label="KS ∝max(c,0)")
            ax.axvline(1, color="r", ls=":", lw=1.2, alpha=0.5, label="δ-пик |c|=1")
            ax.axvline(-1, color="r", ls=":", lw=1.2, alpha=0.5)
            ax.set_title(f"θ={th:.0f}°  {cname}", fontsize=9)
            ax.set_xlabel("c"); ax.set_ylim(0, 3.2); ax.grid(alpha=0.15)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_twist_profile(measure, out_path, title="профиль скрутки τ̃_i"):
    """Средний профиль скрутки ⟨τ̃_i⟩=⟨2z⟩ по цепочке + Tw до/после (R4): как связь
    Tw=const перераспределяет твист (равномерно или к концам)."""
    thetas = sorted(measure.keys())
    fig, ax = plt.subplots(figsize=(8, 4))
    for th in thetas:
        m = measure[th]
        ax.plot(m["tau"], "o-", ms=3, label=f"θ={th:.0f}° (Tw {m['Tw0']:+.3f}→{m['Tw1']:+.3f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("№ связи i вдоль ленты"); ax.set_ylabel("⟨τ̃_i⟩")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_marginals(thetas, p_s, p_t, sig_s, sig_t, out_path):
    """Контроль маргиналов P(s=+), P(t=+) с полосой 0.5 ± 3σ (SPEC §4.2)."""
    deg = np.degrees(thetas)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.5, color="k", lw=1)
    ax.fill_between(deg, 0.5 - 3 * sig_s, 0.5 + 3 * sig_s, alpha=0.15, color="C0",
                    label="0.5 ± 3σ (s)")
    ax.plot(deg, p_s, "o-", ms=4, color="C0", label="P(s=+)")
    ax.plot(deg, p_t, "s-", ms=4, color="C1", label="P(t=+)")
    ax.set_xlabel("θ, градусы")
    ax.set_ylabel("маргинал")
    ax.set_title("Контроль no-signaling: маргиналы концов")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
