# 8. Code, Data & AI methodology (service section)

**Section owner:** Code

## 8.1 Code and data availability

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
`*_prereg.md` before its runs.

Repeatability is exact: each run script fixes its PRNGKey seed protocol, so the same seed
returns the same number, and every figure in this paper can be regenerated from a named commit
by running its committed script — `analysis_ext.py` (d2ext_scaling.png), `analysis_ds2.py`
(ds2_cross.png), `analysis_ds3.py` (ds3_aniso.png, ds3_iso.png), `plot_s1runs_kf.py`
(s1runs_kf.png). Reproducibility — whether a result survives a change of seed — is a separate
property, and we measured it rather than assumed it: a pre-registered audit re-ran the N = 32
and N = 96 cells under twelve fresh base keys, and the cross-seed scatter came out consistent
with the quoted binomial errors (r = 0.88 over eleven repeats of the core cell, r = 1.11 at
N = 96; `seedaudit_report.md`, commit a26f76b).

## 8.2 AI-assisted methodology

The program was run by three parties: an AI architect (Claude — design, pre-registration,
review), an AI executor (Claude Code — implementation, runs, and verification of every drafted
claim against its cited source), and a human owner (decisions and veto). The controls of
Section 3.2 are controls over both AI roles, not over the code alone: pre-registration with
kill-criteria and the raw-before-analysis order bind design and execution to commitments made
before the data existed, which is precisely where a shared preference for a positive result
would otherwise enter. Layered on top, each side checks the other against primary sources —
errors were caught in both directions, and the direction of capture is documented in the
commit history. In the architect's direction: a misattributed number in a discussion draft — a
value belonging to an analytical construction, quoted as though measured — was caught by the
executor's check against the canonical record and never entered the text. In the executor's
direction: a fabricated test count was caught by the executor's own check against the test
runner, and a reproducibility alarm the executor raised on three points was refuted by the
executor's own pre-registered eleven-point audit, which found the quoted errors honest
(Section 5.1). Neither role is reliable unaudited, and neither audits itself by inspection;
what makes the arrangement work is that the commitments are written down first and the checks
run against sources rather than against memory.

## 8.3 Tooling

No proprietary tooling beyond the AI assistants named was used; all analysis code is in the
repository.
