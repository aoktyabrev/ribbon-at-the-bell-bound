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
