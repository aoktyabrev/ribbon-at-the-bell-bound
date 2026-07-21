# The Ribbon at the Bell Bound

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21383667.svg)](https://doi.org/10.5281/zenodo.21383667)
*(concept DOI — all versions)*

Latest release: [v2.0](../../releases/tag/v2.0) — version DOI 10.5281/zenodo.21476426.

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

## Results at a glance

| Result | Status | Commit(s) / source |
|---|---|---|
| Amplitude–form–isotropy trilemma (cycle 1) | measured | `paper/FINAL_v1.md` |
| Factorization of relaxational readout (C2-F) | measured | cycle2 (F_s<6.3e-4) |
| ISO-DYN cosine, S≤2 (third angle of trilemma) | measured | `a043f8f` |
| Quench-glass + structural decorrelation (C2-J) | measured | cycle2 C2-J |
| No-closed-form amplitude A(k_f) (C2-TM) | reduction | cycle2 C2-TM |
| Born-rule selection within the steering class (C3-B) | **theorem** | `968aaa5`, `c152c98` |
| Mechanical steering premise, form-free (C3-B-mech) | measured | `2080bbe`, `1c653b6` |
| Class-M precedence theorem (L2c) | **theorem** | `c8e1bf3` |
| The 1/3-cosine point (local-layer ceiling) | landmark | `a043f8f`, `49e8c1b` |
| Amplitude-seam reduction + trilemma of horns (C3-S) | reduction | `324f097`, `2053106` |
| Before-before anchor: frame-local excluded in nature (L2) | measured | `49e8c1b` |
| Dynamics layer (unitary evolution) | named wall | `sim/cycle4/` |

Only two results carry the status **theorem** (C3-B selection; class-M
precedence). Everything else is measured / reduction / landmark / named wall.

## The bridge

Four registered statements (verbatim opening clauses from the committed Outlook,
`paper/C3_paper_DRAFT_v1.md` §9):

- **RULE** — "wherever a mechanism carries steering … the prohibition of
  superluminal signaling forces the Born exponent p=2 uniquely; the derivation
  is internal, executable, and robust to deformations of the source measure."
- **CEILING** — "mechanisms that resolve outcomes locally … are capped by
  Bell's theorem at S=2, with the 1/3-cosine point as their isotropic-cosine
  ceiling."
- **SEAM** — "the gap between that ceiling and the quantum value is not crossed
  by decorating local mechanisms … it is crossed only by joint resolution of
  the pair."
- **FRAME** — "the surviving corner, preferred-foliation resolution, reproduces
  quantum statistics exactly while rendering its own foliation statistically
  invisible."

The program's bridge statement is, in its own committed words, "not a
unification but an exact division of labor."

## Papers

- **Paper 1** (camera-ready): Zenodo DOI `10.5281/zenodo.21383667`. Source
  `paper/FINAL_v1.md` · PDF [`paper/pdf/main.pdf`](paper/pdf/main.pdf),
  supplement [`paper/pdf/si.pdf`](paper/pdf/si.pdf).
- **Paper 2** (cycle-2 synthesis): in preparation. Source
  `sim/cycle2/C2_synthesis.md` · Technical Report PDF
  [`paper/pdf/c2_synthesis_TR.pdf`](paper/pdf/c2_synthesis_TR.pdf).
- **Paper 3** (cycle-3, RULE/CEILING/SEAM/FRAME): draft v1, in preparation.
  Source `paper/C3_paper_DRAFT_v1.md` · PDF
  [`paper/pdf/c3_draft_v1.pdf`](paper/pdf/c3_draft_v1.pdf) (watermarked DRAFT v1).

## Methodology

- Preregistration before raw data; kill criteria owned by the operator (Artem).
- Raw → commit → analysis; mirror controls; executable prereg↔JSON and
  prereg↔gate assert checklists before every run.
- Claim-discipline: `theorem` only with written proof + battery + review;
  `conjecture` otherwise; metric = the model observable; analytics beside numerics.
- **False stops and retractions are public and are a feature, not dirty
  laundry** — e.g. the quadrature-gate false stop caught and corrected without
  loosening any registered criterion (`ccf5af6`); the §2.3 Tsirelson over-claim
  retracted after a battery rediscovered the PR-box fact (`2053106`); the M0
  finite-T design gap reclassified as calibration (`0a6c51e`).

## Open walls / Cycle 4

- **C4-GHZ** flagship (two tracks) — priors: G-T (N-event class-M theorem;
  Mermin from global disorder) **0.7**; G-M (three-end geometric law
  reproducing GHZ with invisible precedence) **0.35**; named wall
  ("bipartite-complete, tripartite-locked" + exact breaking property) **0.45**
  (`990cb99`, `12cb5d6`). Critique triage: `sim/cycle4/C4_critique_triage.md`.
- **Dynamics layer** — unitary time evolution is outside the class of
  statistical-measurement models; a named boundary, not a justification.
- **C3-G** — ground state of the stiff frustrated chain (parked).
- **C2-TM** — the relaxational amplitude has no closed form (observation).

## Roles

The operator and owner of the kill criteria is Artem. The "architect" and
"executor" are AI roles (Claude) run in an adversarial scheme: the executor
runs adversarial passes against the architect's theorem drafts, while the
architect reviews the executor's analytics and gates; theorem status is granted
only after a clean pass plus operator review. This is stated plainly — see
Methodology; it is neither hidden nor foregrounded.

## Repository map

- `paper/` — papers, sections, PDFs, reviews (paper 1 final; paper 3 draft).
- `sim/phase_D/` — core engine (band4d, measurement, ribbon_sim) + `results/` raw JSON.
- `sim/cycle2/` — cycle-2 campaigns (C2-F/TM/ISO/A/L/R/J) + synthesis.
- `sim/cycle3/` — cycle-3 (C3-B theorem, C3-L class-M theorem, C3-S seam, C3-B-mech).
- `sim/cycle4/` — cycle-4 scaffold (C4-GHZ G-T/G-M drafts, critique triage).
- `notes/` — historical notes and figures.
