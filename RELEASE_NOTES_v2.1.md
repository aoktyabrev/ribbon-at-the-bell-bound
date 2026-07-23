# Release v2.1 — "The tripartite mirror, and a submission"

**Version DOI: 10.5281/zenodo.21505219**
(concept DOI `10.5281/zenodo.21383667` — all versions).

A three-cycle, preregistered research program on the boundary between classical
mechanisms and quantum correlations. Cycle 1 mapped the walls (the
amplitude–form–isotropy trilemma of a geometric hidden-connection model).
Cycle 2 measured why the walls stand (factorization of relaxational readout;
rule-dial; quench-glass; structural decorrelation). Cycle 3 crossed to the
reconstruction side: an internal, executable derivation of the Born rule from
no-signaling (with a mechanically realized, form-free steering premise), and a
theorem for the class of all outcome-definite resolution mechanisms — S>2
requires a shared, statistically invisible precedence structure. Every claim in
this README carries the commit that proves it; every campaign ran under
preregistered kill criteria, and the false stops are part of the public record.

## What's new since v2.0

- **Mechanical steering premise, form-free (C3-B-mech)** — annealed,
  field-magnetized, fast-frozen elastic source; D>0 in the 3rd and 5th moments,
  h=0 control null; no closed form needed (`2080bbe`, `1c653b6`).
- **Cycle 4 — the tripartite mirror.**
  - *Value theorem at N=3:* Mermin M₃>2 achievable in class M iff ≥1 space-like
    pair is ordered; one ordered pair suffices for the algebraic maximum
    (`840602b`, β2 `7b8c371`).
  - *Tripartite mirror:* the seam (GM-F2, GHZ signature at the classical bound,
    predicted 0.45; `99205c3`) and its crossing (GM-F2j, complete GHZ signature
    at the Mermin settings from parity-in-geometry + one ordered pair; `ed8e950`).
  - *Binary block-size tier:* disjoint ordered pairs cap the Mermin-Klyshko value
    at the 2-producible ceiling for N≥4 and at the Svetlichny bound (S₃=4,
    `ca6d220`); only a full precedence block attains the top (algebraic 4.0 at
    N=4, above quantum 2^{(N−1)/2}; m=4 `cb76cbd`).
  - *Schedule-invisibility theorem (class M):* the linear extension of the
    precedence structure is statistically invisible; the structure is certified
    by achieved tiers (v2, clean second pass; `bb0cec6`, battery `d612214`).
- **Paper 3 restructured for submission (v3)** — compact method frame as new §2
  (prereg, kill criteria, bitwise reproduction in two environments, retraction
  register as one table); cycle-4 Outlook; `paper/pdf/c3_draft_v3.pdf`.
- **Registered cycle-4 retractions** (self-correction register, a feature):

  | Over-claim | Caught by | Resolution | Commit |
  |---|---|---|---|
  | k-pair "climbing ladder" (N=4 beats at k=2) | kinship vs standard Mermin-Klyshko | operator artifact; k=1=k=2=2.000 | `ff28341` |
  | "full GHZ statistics" (GM-F2j) | Svetlichny genuine-N check | scope → "GHZ signature at the Mermin settings" | `60e3246` |
  | "45°-tail gains (√2)^{N−2}" | executor pencil before run | deterministic tail is classical; √2 needs superposition | `37b9af8` |

## Assets
- `c3_draft_v3.pdf` — paper 3 (cycle-3), submission v3.
- `c2_synthesis_TR.pdf` — cycle-2 synthesis, technical report.
- `main.pdf`, `si.pdf` — paper 1 (camera-ready) + supplement.

## Reproducibility
Every quantitative claim carries a commit hash; batteries are single numpy (or
JAX/GPU) scripts run under preregistered kill criteria; the cycle reproduces
bitwise. Repository: <https://github.com/aoktyabrev/ribbon-at-the-bell-bound>.
