# Supplementary: AI-assisted methodology (detailed)

*Supplementary to Section 9.2 of the main text. Moved here per journal-policy formatting of
the AI disclosure; the main text carries the disclosure statement, this file carries the
working arrangement and its evidence.*

## SI-M.1 The three-party arrangement and two-directional self-correction

The program was run by three parties: an AI architect (Claude — design, pre-registration,
review), an AI executor (Claude Code — implementation, runs, and verification of every drafted
claim against its cited source), and a human owner (decisions and veto). The controls of
Section 4.1 are controls over both AI roles, not over the code alone: pre-registration with
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
(Section 6.1). Neither role is reliable unaudited, and neither audits itself by inspection;
what makes the arrangement work is that the commitments are written down first and the checks
run against sources rather than against memory.
