"""Fig. 7 (paper 3): the trilemma with the ISO-DYN edge, protocols named.

All three property pairs are populated here, which is exactly why every edge
must carry its PROTOCOL. Two of the edges isotropize and one does not, and the
two isotropizations are different operations. Without the protocol line the
DS3 edge ("honest isotropy forces the triangle") and the ISO-DYN edge ("a
cosine at suppressed amplitude") read as a contradiction. They are not: the
settings enter the dynamics in one and never touch it in the other.

  amplitude + isotropy, NO form   -- DS3 honest isotropization.
      Haar R applied POST HOC, in analysis, to the frozen theta=0 source;
      settings drawn independently of lambda => a shared-lambda local model,
      and Bell's theorem forces the triangular signature. 0fb5452.

  amplitude + form, NO isotropy   -- DS3 anisotropy map.
      Fixed preparation; the clamp axis is tilted by alpha off the privileged
      axis e-hat and A(alpha) is read. 0fb5452.

  isotropy + form, NO amplitude   -- ISO-DYN (cycle 2).
      Per-replica rigid Haar rotation of the PREPARATION before relaxation;
      (a, b) lab-fixed and entering the relaxation as clamps. The settings act
      on the dynamics, so this is NOT a shared-lambda model with independent
      settings, and a cosine is available -- paid for with amplitude.
      a043f8f, prereg 77f734c, raw 3fd3e3d.

Two caveats from the ISO-DYN pre-registration are load-bearing and are drawn on
the figure, not left to the caption alone:

  (1) the isotropy is Haar-AVERAGING of an anisotropic dynamical response, not
      a symmetry of the model (the prereg's own words: "не операция симметрии");
  (2) ISO-DYN BRACKETS the 1/3-cosine point, it does not land on it.

Run: PYTHONPATH=src sim/.venv/bin/python cycle3/plot_trilemma_isodyn.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "phase_D", "results")
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, "DS3_analysis.json"), encoding="utf-8") as f:
    ds3 = json.load(f)
RHO_TRI = {k: ds3["iso"][f"kf{k}.0"]["rho_triangle"] for k in (1, 4)}
A_ANISO = {k: (ds3["aniso"][f"kf{k}.0"]["A"][0],
               ds3["aniso"][f"kf{k}.0"]["A"][-1]) for k in (1, 4)}

with open(os.path.join(RES, "C2ISO_analysis.json"), encoding="utf-8") as f:
    iso = json.load(f)["cells"]
RHO_COS = {k: iso[f"kf{k}.0"]["rho_cos"] for k in (1, 4)}
DAICC = {k: iso[f"kf{k}.0"]["dAICc_cos_minus_tri"] for k in (1, 4)}
S_ISO = {k: iso[f"kf{k}.0"]["H_I2"]["S"] for k in (1, 4)}
MODEL_FREE = iso["kf4.0"]["H_I0"]["model_free"]

V = {
    "amp": np.array([0.0, 1.05]),
    "iso": np.array([-1.30, -0.62]),
    "form": np.array([1.30, -0.62]),
}
CENTER = np.array([0.0, -0.075])

DS3C = "#1f4e79"      # phase-D protocol
ISOC = "#0b7a5a"      # cycle-2 protocol
TARGET = "#6a3d9a"

fig, ax = plt.subplots(figsize=(11.6, 8.4))

for a, b, col in (("amp", "iso", DS3C), ("amp", "form", DS3C),
                  ("iso", "form", ISOC)):
    ax.plot(*zip(V[a], V[b]), color=col, lw=2.8, zorder=3,
            solid_capstyle="round")

VLAB = {
    "amp": ("unit amplitude   $\\rho = 1$", "center", "bottom", (0.0, 0.10)),
    "iso": ("isotropy", "center", "top", (-0.10, -0.16)),
    "form": ("cosine form", "center", "top", (0.10, -0.16)),
}
for k, (lab, ha, va, off) in VLAB.items():
    ax.plot(*V[k], marker="o", ms=13, color="#333333", zorder=6)
    ax.text(V[k][0] + off[0], V[k][1] + off[1], lab, ha=ha, va=va,
            fontsize=13.5, fontweight="bold", zorder=6)

ax.plot(CENTER[0], CENTER[1] + 0.16, marker="*", ms=28, color=TARGET, zorder=7)
ax.text(CENTER[0], CENTER[1] - 0.02,
        "quantum target\nisotropic cosine, $\\rho = 1$, $S = 2\\sqrt{2}$\n"
        "$\\bf{no\\ edge\\ reaches\\ it}$",
        ha="center", va="top", fontsize=10.6, color=TARGET, zorder=7,
        linespacing=1.5)


def box(xy, title, protocol, result, colour, face="white"):
    txt = (f"$\\bf{{{title}}}$\n"
           f"$\\it{{protocol:}}$ {protocol}\n"
           f"{result}")
    ax.text(xy[0], xy[1], txt, ha="center", va="center", fontsize=9.3,
            color=colour, zorder=8, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=face,
                      edgecolor=colour, linewidth=1.5, alpha=0.97))


box(np.array([-2.62, 0.38]), "DS3\\ \\ honest\\ isotropization",
    "Haar $R$ applied post hoc, in analysis,\n"
    "to the frozen $\\theta=0$ source; settings\n"
    "independent of $\\lambda$ $\\Rightarrow$ shared-$\\lambda$ model",
    "has: amplitude + isotropy — lacks: form\n"
    f"triangle $E=-\\rho(1-2\\theta/\\pi)$, $\\rho={RHO_TRI[1]:.3f}/{RHO_TRI[4]:.3f}$\n"
    "$S = 2\\rho$   (0fb5452)", DS3C)

box(np.array([2.62, 0.38]), "DS3\\ \\ anisotropy\\ map",
    "fixed preparation; the clamp axis is\n"
    "tilted by $\\alpha$ off the privileged axis $\\hat{e}$",
    "has: amplitude + form — lacks: isotropy\n"
    f"$A(\\alpha)$: {A_ANISO[1][0]:.3f}$\\to${A_ANISO[1][1]:.3f} "
    f"($k_f\\times1$), {A_ANISO[4][0]:.3f}$\\to${A_ANISO[4][1]:.3f} ($k_f\\times4$)\n"
    "cosine only along $\\hat{e}$   (0fb5452)", DS3C)

box(np.array([0.0, -1.72]), "ISO\\text{-}DYN\\ \\ (cycle\\ 2)",
    "per-replica rigid Haar rotation of the\n"
    "$\\bf{preparation\\ before\\ relaxation}$; $(a,b)$ lab-fixed,\n"
    "$\\bf{entering\\ the\\ relaxation\\ as\\ clamps}$",
    "has: isotropy + form — lacks: amplitude\n"
    f"cosine wins decisively: $\\Delta$AICc $={DAICC[1]:.1f}$ / ${DAICC[4]:.1f}$\n"
    f"$\\rho_{{\\rm cos}} = {RHO_COS[1]:.3f} / {RHO_COS[4]:.3f}$,  "
    f"$S = 2\\sqrt{{2}}\\rho = {S_ISO[1]:.3f} / {S_ISO[4]:.3f} \\leq 2$\n"
    "(a043f8f; prereg 77f734c, raw 3fd3e3d)", ISOC, face="#f0f8f5")

# --- the two pre-registered caveats, on the figure --------------------------
ax.text(-3.32, -1.72,
        "$\\bf{caveat\\ 1}$\nthe isotropy is Haar-$\\it{averaging}$\n"
        "of an anisotropic dynamical\nresponse — $\\bf{not\\ a\\ symmetry}$\n"
        "$\\bf{of\\ the\\ model}$ (prereg 77f734c)",
        ha="center", va="center", fontsize=8.8, color=ISOC, zorder=8,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf2",
                  edgecolor=ISOC, linewidth=1.1, linestyle="--", alpha=0.97))

ax.text(3.32, -1.72,
        "$\\bf{caveat\\ 2}$\nISO-DYN $\\bf{brackets}$ the 1/3-cosine\n"
        f"point, it does not land on it:\n"
        f"$\\rho_{{\\rm cos}}={RHO_COS[4]:.3f}$, $\\rho_{{\\rm model\\text{{-}}free}}={MODEL_FREE:.3f}$\n"
        "straddle 1/3 within $\\sim2\\sigma$",
        ha="center", va="center", fontsize=8.8, color=ISOC, zorder=8,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffdf2",
                  edgecolor=ISOC, linewidth=1.1, linestyle="--", alpha=0.97))

ax.text(0.0, 1.86,
        "Each edge buys two properties and pays with the third.\n"
        "The two isotropizations are $\\bf{different\\ protocols}$, not a contradiction:\n"
        "in ISO-DYN the settings act on the dynamics; in DS3 they never touch it.",
        ha="center", va="center", fontsize=11.2, color="#333333",
        linespacing=1.6)

ax.set_xlim(-4.15, 4.15)
ax.set_ylim(-2.35, 2.25)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
out = os.path.join(FIG, "c3_trilemma_isodyn.png")
plt.savefig(out, dpi=130)
plt.close()

print(f"ISO-DYN rho_cos = {RHO_COS[1]:.4f} / {RHO_COS[4]:.4f}  "
      f"dAICc = {DAICC[1]:.2f} / {DAICC[4]:.2f}  S = {S_ISO[1]:.4f} / {S_ISO[4]:.4f}")
print(f"model_free(kf4) = {MODEL_FREE}   DS3 rho_tri = "
      f"{RHO_TRI[1]:.4f} / {RHO_TRI[4]:.4f}")
print(f"-> {out}")
