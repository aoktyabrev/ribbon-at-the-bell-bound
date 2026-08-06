"""Fig. 5 (paper 3, §5): the CHSH landscape on one axis.

Every point is a CHSH (N=2) value, so they share one axis. Two classes:

  ANALYTIC / LITERATURE -- the 1/3-cosine ceiling 2sqrt(2)/3 (lever rule,
  exact), the Bell bound 2, the Tsirelson bound 2sqrt(2), and the algebraic
  maximum 4 (the PR box).

  MEASURED -- the phase-D relaxation family S = 2*rho at k_f x {1, 4}, read
  off the frozen DS3 analysis.

CAUTION (operator hygiene, per the cycle-4 climbing-ladder retraction ff28341):
the "4.0" here is the CHSH algebraic maximum -- the PR box, N=2. It is NOT the
Mermin-Klyshko algebraic value 4.0 at N=4 quoted in the Outlook. The two numbers
coincide but belong to different operators and must never share an axis by
accident. This figure is CHSH only.

Raw inputs (no number is typed in by hand except literature/analytic constants):
  DS3_analysis.json  iso/kf{1,4}/rho_triangle   -> S = 2*rho    (commit 0fb5452)
  C3L_L2.json        lever rule E = cos(theta)/3 -> S = 2sqrt(2)/3 (commit 49e8c1b)

Run: PYTHONPATH=src sim/.venv/bin/python cycle3/plot_chsh_landscape.py
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

SQRT2 = np.sqrt(2.0)

# --- measured: phase-D relaxation family, S = 2*rho (triangular LHV) ---------
with open(os.path.join(RES, "DS3_analysis.json"), encoding="utf-8") as f:
    ds3 = json.load(f)["iso"]
S_meas = {k: 2.0 * ds3[f"kf{k}.0"]["rho_triangle"] for k in (1, 4)}

# --- analytic / literature landmarks ----------------------------------------
CEILING = 2.0 * SQRT2 / 3.0      # 1/3-cosine point, lever rule (exact)
BELL = 2.0                       # [B64] / [CHSH69]
TSIRELSON = 2.0 * SQRT2          # [C80]
ALGEBRAIC = 4.0                  # PR box [PR94] -- CHSH algebraic maximum

# Layout lanes (y): analytic/literature ABOVE the axis, measured BELOW it,
# so the crowded interval [0.7, 1.7] never stacks two classes at one height.
XMAX = 4.28
TIERS = [
    (0.0, BELL, "local / shared-$\\lambda$ tier", "#eef2f7"),
    (BELL, TSIRELSON, "quantum tier", "#e6f0e6"),
    (TSIRELSON, XMAX, "post-quantum tier (no-signaling)", "#f7efe6"),
]

fig, ax = plt.subplots(figsize=(10.4, 4.6))

for lo, hi, label, fc in TIERS:
    ax.axvspan(lo, hi, color=fc, zorder=0)
    ax.text((lo + min(hi, XMAX)) / 2, 2.16, label, ha="center", va="center",
            fontsize=9, color="#5a6472", zorder=1)

ax.axhline(0, color="#333333", lw=1.5, zorder=2)

# --- analytic + literature landmarks, above the axis ------------------------
# (x, stem top, label, provenance) -- stems staggered so labels never collide
ABOVE = [
    (CEILING, 1.02, "#1f4e79", "1/3-cosine ceiling\n$S=2\\sqrt{2}/3\\approx0.943$",
     "lever rule, exact\nC3L_L2.json, 49e8c1b"),
    (BELL, 0.52, "#333333", "Bell bound\n$S=2$", "[B64], [CHSH69]"),
    (TSIRELSON, 1.02, "#333333", "Tsirelson\n$S=2\\sqrt{2}\\approx2.828$", "[C80]"),
    (ALGEBRAIC, 0.52, "#333333", "algebraic max (PR box)\n$S=4$",
     "[PR94]; §9, 2053106"),
]
for x, top, col, lab, prov in ABOVE:
    ax.plot([x, x], [0, top], color=col, lw=1.1, ls="--", zorder=3)
    ax.plot([x], [0], marker="D" if col != "#333333" else "|", ms=9,
            color=col, zorder=6)
    # provenance sits closest to the stem, label above it: reading top-down
    # gives label -> source, not source -> label.
    ax.text(x, top + 0.05, prov, ha="center", va="bottom", fontsize=7.4,
            color=col, style="italic", alpha=0.85, zorder=4)
    ax.text(x, top + 0.10 + 0.20 * prov.count("\n") + 0.20, lab,
            ha="center", va="bottom", fontsize=9.5, color=col, zorder=4)

# --- the seam: ceiling -> Tsirelson -----------------------------------------
ax.annotate("", xy=(TSIRELSON, 0.24), xytext=(CEILING, 0.24),
            arrowprops=dict(arrowstyle="<->", color="#6a3d9a", lw=1.6), zorder=6)
ax.text((CEILING + TSIRELSON) / 2, 0.28, "amplitude seam",
        ha="center", va="bottom", fontsize=9.5, color="#6a3d9a", zorder=6)

# --- measured relaxation family, below the axis -----------------------------
MEAS = "#a63603"
for k, depth in ((1, -0.62), (4, -1.02)):
    x = S_meas[k]
    ax.plot([x, x], [0, depth], color=MEAS, lw=1.0, zorder=3)
    ax.plot([x], [0], marker="o", ms=8, color=MEAS, zorder=6)
    ax.text(x, depth - 0.05, f"$k_f\\times{k}$:  $S=2\\rho={x:.2f}$",
            ha="center", va="top", fontsize=9.5, color=MEAS, zorder=6)
ax.annotate("", xy=(S_meas[4], -0.28), xytext=(S_meas[1], -0.28),
            arrowprops=dict(arrowstyle="->", color=MEAS, lw=1.4), zorder=5)
ax.text((S_meas[1] + S_meas[4]) / 2, -0.24,
        "measured family (stiffness)", ha="center", va="bottom",
        fontsize=8.6, color=MEAS, zorder=6)
ax.text(S_meas[4], -1.30, "DS3_analysis.json, 0fb5452",
        ha="center", va="top", fontsize=7.4, color=MEAS,
        style="italic", alpha=0.85, zorder=6)

ax.set_xlim(-0.06, XMAX)
ax.set_ylim(-1.75, 2.40)
ax.set_xlabel("CHSH value $S$   (N = 2, one operator throughout)")
ax.set_xticks([0, 1, 2, 3, 4])
ax.get_yaxis().set_visible(False)
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_position(("data", -1.75))

plt.tight_layout()
out = os.path.join(FIG, "c3_chsh_landscape.png")
plt.savefig(out, dpi=130)
plt.close()

print(f"S(k_f x1) = 2*rho = {S_meas[1]:.4f}   S(k_f x4) = {S_meas[4]:.4f}")
print(f"ceiling 2sqrt2/3 = {CEILING:.4f}  Bell = {BELL}  "
      f"Tsirelson = {TSIRELSON:.4f}  algebraic = {ALGEBRAIC}")
print(f"-> {out}")
