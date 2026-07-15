# 9. Code, Data & AI methodology (service section)

**Section owner:** Code

## 9.1 Code and data availability

All code, data, and pre-registrations are in the repository: the phase A–C toolkit in
`sim/src/ribbon_sim/`, the phase-D model and campaigns in `sim/phase_D/`, and the frozen raw
measurements in `sim/phase_D/results/*.json`.

Every stage whose result required a fit, a flip, or a model comparison committed its raw
measurements before any analysis touched them, as a `*-raw` commit preceding a `*-analysis`
commit: D2 (7a2f18d → 9d7b8e9), D2-ext (624b628 → 2784edf), S1-runs (704e29c → a9cef7b),
DS2 (a9f6cfe → f928dd4), DS3 (3d8dd67 → 0fb5452), and the seed audit (2b946ae
pre-registration → ecff715 raw → a26f76b analysis). The two infrastructure stages, D0 and D1,
produced deterministic validations rather than fitted numbers and are single commits
(f25694d, 81cc7da). Each campaign's hypotheses and kill-criteria are fixed in a committed
`*_prereg.md`, and what the record proves is graded (Section 4.1): no pre-registration was
ever edited after entering it (eight files, one commit each); all entered it before the
analysis they govern; and for the seed audit — the one campaign whose pre-registration was
committed as its own step (2b946ae) — before the runs as well.

Repeatability is exact: each run script fixes its PRNGKey seed protocol, so the same seed
returns the same number, and every figure in this paper can be regenerated from a named commit
by running its committed script — `analysis_ext.py` (d2ext_scaling.png), `analysis_ds2.py`
(ds2_cross.png), `analysis_ds3.py` (ds3_aniso.png, ds3_iso.png), `plot_s1runs_kf.py`
(s1runs_kf.png). Reproducibility — whether a result survives a change of seed — is a separate
property, and we measured it rather than assumed it: a pre-registered audit re-ran the N = 32
and N = 96 cells under twelve fresh base keys, and the cross-seed scatter came out consistent
with the quoted binomial errors (r = 0.88 over eleven repeats of the core cell, r = 1.11 at
N = 96; `seedaudit_report.md`, commit a26f76b).

## 9.2 Use of generative AI

Generative AI tools (Anthropic Claude) were used for research planning and experimental
design, software implementation, code review, source checking, and manuscript drafting. The
author made all scientific decisions, verified the reported results, and accepts full
responsibility for the manuscript.

The detailed working arrangement — the three-party division of roles, the pre-registration and
raw-before-analysis order as controls over both AI roles, and the record of errors caught in
both directions — is given in `SI_methodology.md`.

## 9.3 Tooling

No proprietary tooling beyond the AI assistants named was used; all analysis code is in the
repository.
