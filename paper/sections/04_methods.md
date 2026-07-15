# 4. Methods

**Section owner:** Code

## 4.1 Pre-registration and controls

Every campaign carries a pre-registration file fixing hypotheses, quantitative expectations,
and kill-criteria. The commit record proves three properties at three strengths: no
pre-registration was ever edited after entering the record (all eight files: one commit each);
every pre-registration entered the record before the analysis it governs (raw-data commits
precede analysis commits throughout); and for the final campaign, the seed audit, the
pre-registration was committed before the runs themselves. We claim the stronger "registered
before runs" as working practice throughout; the commit record proves it directly for the last
campaign and proves "registered before analysis, never edited" for all. A representative entry
(D0, hypothesis D-H2; translated from the Russian original)
reads: "ℤ₂ parity is exactly conserved
on accepted trajectories, and the singular-rejection fraction → 0 as dt → 0; kill: if the
rejection fraction does not fall with dt, the parity is an artifact of the filter"
(D0_prereg.md). Registering the failure condition in advance is what let a null result count
as a result rather than as a disappointment to be explained away.

The analysis order was strict: every stage whose result required a fit, a flip, or a model
comparison committed its raw measurements before any analysis touched them, as a `*-raw`
commit preceding a `*-analysis` commit; the two infrastructure stages, D0 and D1, produced
deterministic validations rather than fitted numbers and are single commits (the commit chain
is listed in Section 9.1). Every
campaign carried controls: an unbiased preparation sampler (boundary axes drawn uniformly on
S², sector preparation without bias, so the branch is never chosen in advance and the basin
decides; measurement.py), mirror pairs (a, b) ↔ (−a, −b), block convergence of the reported
observable, an explicit DEGENERATE class for undetermined basins, and — for the central
scaling claim — a pre-registered cross-seed audit of the quoted errors (Section 4.2). A
standing kill-criterion
treated any CHSH > 2 in a manifestly local model as a protocol error to be audited before
interpretation; it fired twice, and both times caught a method artifact rather than physics
(Section 7).

This machinery was not incidental to the working arrangement but demanded by it. The program
was run under an explicit division of roles between generative-AI tooling — design and
pre-registration on one side, implementation, runs, and verification on the other — and the
human author, who made all decisions and held veto (Section 9.2 and the Supplementary
methodological note). Pre-registration, kill-criteria,
and the raw-before-analysis order are controls over both AI roles: they bind the design and
the execution to commitments made before the data existed, which is precisely the point at
which the shared hope for a positive result would otherwise leak into the pipeline.

## 4.2 Repeatability and reproducibility

Two distinct properties are worth separating. Repeatability is exact: every stage fixes a
PRNGKey seed protocol and commits its raw data, so the same seed returns the same number and
any figure can be regenerated from a named commit. Reproducibility — whether a result
survives a change of seed — is a separate question, and we measured it rather than assumed
it: a dedicated seed audit re-ran the N = 32 and N = 96 cells under twelve fresh base keys
and found the cross-seed scatter consistent with the quoted binomial errors (r = 0.88 and
1.11; Section 6.1, seed audit commit a26f76b).

The phase history of the program (phases A–D) and the implementation detail — frame algebra,
test coverage by invariant, numerical precision — are given in `SI/SI_program_history.md`.
