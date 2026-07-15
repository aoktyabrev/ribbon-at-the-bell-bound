# notes/

Working documents that are not part of the paper but are part of its provenance. Nothing here
is a result; read the paper (`paper/`) for those, and the reports under `sim/phase_D/` for the
measurements behind them.

## ai-collaboration/

The role contexts that drove the three-party arrangement described in the paper's
Supplementary methodology (`paper/SI/SI_methodology.md`): an AI architect (design,
pre-registration, review), an AI executor (implementation, runs, verification against
sources), and the human author (decisions and veto).

These are primary documents of that methodology, not scaffolding: `ARCHITECT_CONTEXT.md` and
`CLAUDE_CODE_CONTEXT.md` are the briefs each AI role was started from, and they state the
working rules — pre-registration is not edited after a run, stop on a red control, do not tune
parameters without recording the decision, escalate ambiguity rather than guess. The paper
claims that these controls governed both AI roles; these files are what that claim refers to.
They are published for the same reason the withdrawn CHSH values are struck through rather
than deleted.

Note: they were written for working sessions, not for readers. They are in Russian, they
describe the state of the program at the moment each session began (so parts are outdated —
e.g. test counts and phase status), and they were never revised for publication.

## historical/

Material from the analytical layer and early phases that is superseded or was never tested.
Kept because the paper's account of its own history should be checkable.

- `R1_theory_prediction.md`, `R1_theory_prediction.png`, `kappa_theory.py` — **historical,
  superseded.** A theoretical prediction for R1 fixed on 2026-07-06 before the data were
  opened, with the rigid-limit analytics behind it. The R1 campaign then hit a convergence
  wall and the paranoia protocol (see `sim/results/decisions.md`); the prediction was not
  vindicated in the form written. `R1_theory_prediction.md` references a data file
  `kappa_curves.npy` that is **not in the repository** — the figure can be regenerated from
  `kappa_theory.py`.
- `boundary_opacity_hypothesis_v2.md` — **historical, untested.** A speculative framing
  ("calibrated causal opacity of the boundary") from the analytical layer. It is not cited by
  the paper, no result depends on it, and it was never subjected to a pre-registered test.

These files are in Russian.
