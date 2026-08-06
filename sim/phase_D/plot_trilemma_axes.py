"""Fig. 6 (paper 1, §6.5): the three axes of failure.

The quantum target -- an isotropic cosine at unit amplitude, CHSH = 2sqrt(2) --
lies outside the measured family on three independent axes AT ONCE: amplitude,
form, and isotropy. This figure is that sentence, drawn.

Topology, and why it is not a populated triangle: phase D populates only TWO of
the three property pairs.

  amplitude + isotropy, NO cosine form   -- 6.4, honest isotropization
  amplitude + cosine form, NO isotropy   -- 6.3, anisotropy map
  isotropy + cosine form, NO amplitude   -- EMPTY in phase D

The third corner is left empty and labelled as empty. Drawing a mechanism there
would be drawing a mechanism this paper does not have: the pair (isotropy +
form) is realised only by cycle-2 ISO-DYN, which is outside this paper's
evidence base. The emptiness is part of the result, not a gap in the layout.

Every number is read from frozen phase-D analysis JSON. Commits cited in the
caption are exactly 0fb5452 (DS3), 2784edf (D2-ext), f928dd4 (DS2), a9cef7b
(S1-runs) -- no other provenance enters this figure.

Run: PYTHONPATH=src sim/.venv/bin/python phase_D/plot_trilemma_axes.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, "DS3_analysis.json"), encoding="utf-8") as f:
    ds3 = json.load(f)
RHO = {k: ds3["iso"][f"kf{k}.0"]["rho_triangle"] for k in (1, 4)}       # 6.4
A_ANISO = {k: (ds3["aniso"][f"kf{k}.0"]["A"][0],
               ds3["aniso"][f"kf{k}.0"]["A"][-1]) for k in (1, 4)}      # 6.3

# --- geometry ---------------------------------------------------------------
V = {                                   # property vertices
    "amp": np.array([0.0, 1.05]),       # unit amplitude
    "iso": np.array([-1.30, -0.62]),    # isotropy
    "form": np.array([1.30, -0.62]),    # cosine form
}
CENTER = np.array([0.0, -0.075])

POP = "#1f4e79"      # populated edge
EMPTY = "#b03030"    # empty edge
TARGET = "#6a3d9a"

fig, ax = plt.subplots(figsize=(10.6, 7.2))

# --- edges ------------------------------------------------------------------
# populated: amplitude+isotropy (6.4), amplitude+form (6.3)
for a, b in (("amp", "iso"), ("amp", "form")):
    ax.plot(*zip(V[a], V[b]), color=POP, lw=2.6, zorder=3, solid_capstyle="round")
# empty: isotropy+form
ax.plot(*zip(V["iso"], V["form"]), color=EMPTY, lw=2.0, ls=(0, (5, 5)),
        zorder=3)

# --- property vertices: each IS an axis of failure, so the vertex label
#     carries both the property and the measured way the target misses it ----
VLAB = {
    "amp": ("unit amplitude   $\\rho = 1$",
            f"measured $\\rho \\leq {RHO[4]:.2f}$; plateau $A = 0.363$,\n"
            "flat in stiffness (6.2; 2784edf, f928dd4, a9cef7b)",
            "center", "bottom", (0.0, 0.10)),
    "iso": ("isotropy",
            "the cosine survives only along $\\hat{e}$:\n"
            f"$A(\\alpha)\\!:\\,{A_ANISO[4][0]:.3f} \\to {A_ANISO[4][1]:.3f}$ "
            "at $\\alpha = \\pi/2$ (6.3)",
            "center", "top", (-0.10, -0.16)),
    "form": ("cosine form",
             "honest isotropy forces the triangle:\n"
             "$E = -\\rho(1 - 2\\theta/\\pi)$, not a cosine (6.4)",
             "center", "top", (0.10, -0.16)),
}
for k, (lab, sub, ha, va, off) in VLAB.items():
    ax.plot(*V[k], marker="o", ms=13, color="#333333", zorder=6)
    x, y = V[k][0] + off[0], V[k][1] + off[1]
    if va == "bottom":
        ax.text(x, y, lab, ha=ha, va="bottom", fontsize=13,
                fontweight="bold", zorder=6)
        ax.text(x, y + 0.17, sub, ha=ha, va="bottom", fontsize=8.8,
                color="#5a6472", zorder=6, linespacing=1.4)
    else:
        ax.text(x, y, lab, ha=ha, va="top", fontsize=13,
                fontweight="bold", zorder=6)
        ax.text(x, y - 0.17, sub, ha=ha, va="top", fontsize=8.8,
                color="#5a6472", zorder=6, linespacing=1.4)

# --- the quantum target, at the centre: needs all three ---------------------
ax.plot(CENTER[0], CENTER[1] + 0.16, marker="*", ms=28, color=TARGET, zorder=7)
ax.text(CENTER[0], CENTER[1] - 0.02,
        "quantum target\nisotropic cosine, $\\rho=1$, $S=2\\sqrt{2}$\n"
        "$\\bf{outside\\ the\\ family\\ on\\ all\\ three\\ axes}$",
        ha="center", va="top", fontsize=10.4, color=TARGET, zorder=7,
        linespacing=1.5)

# --- mechanism boxes, OUTSIDE the two populated edges -----------------------
def box(xy, title, body, colour, face="white", ls="-"):
    ax.text(xy[0], xy[1], f"$\\bf{{{title}}}$\n{body}", ha="center",
            va="center", fontsize=9.6, color=colour, zorder=8,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=face,
                      edgecolor=colour, linewidth=1.4, linestyle=ls,
                      alpha=0.97))

box(np.array([-2.08, 0.34]), "6.4\\ \\ honest\\ isotropization",
    "has: amplitude + isotropy\nlacks: cosine form\n"
    f"$\\rho = {RHO[1]:.3f}\\,/\\,{RHO[4]:.3f}$,  $S = 2\\rho$\n0fb5452", POP)

box(np.array([2.08, 0.34]), "6.3\\ \\ anisotropy\\ map",
    "has: amplitude + cosine form\nlacks: isotropy\n"
    f"$A$: {A_ANISO[1][0]:.3f}$\\,\\to\\,${A_ANISO[1][1]:.3f} ($k_f\\times1$)\n"
    f"{A_ANISO[4][0]:.3f}$\\,\\to\\,${A_ANISO[4][1]:.3f} ($k_f\\times4$)\n0fb5452",
    POP)

# --- the empty corner, labelled as empty ------------------------------------
mid_empty = (V["iso"] + V["form"]) / 2
ax.plot([mid_empty[0], mid_empty[0]], [mid_empty[1], -1.32],
        color=EMPTY, lw=1.1, ls=(0, (3, 3)), zorder=3)
box(np.array([mid_empty[0], -1.52]),
    "empty\\ in\\ phase\\ D",
    "isotropy + cosine form, no amplitude\n"
    "no mechanism of this paper populates it", EMPTY,
    face="#fdf3f3", ls="--")

ax.text(0.0, 1.72,
        "No single knob moves the ribbon toward the target:\n"
        "each mechanism buys two properties and pays with the third.",
        ha="center", va="center", fontsize=11.5, color="#333333",
        linespacing=1.5)

ax.set_xlim(-3.15, 3.15)
ax.set_ylim(-1.95, 2.05)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
out = os.path.join(FIG, "trilemma_axes.png")
plt.savefig(out, dpi=130)
plt.close()

print(f"rho (6.4) = {RHO[1]:.4f} / {RHO[4]:.4f}")
print(f"A(alpha) (6.3) kf1 {A_ANISO[1]}  kf4 {A_ANISO[4]}")
print(f"-> {out}")
