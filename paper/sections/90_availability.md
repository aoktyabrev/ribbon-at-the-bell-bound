# 8. Code, Data & AI methodology (service section)  [STUB]

**Section owner:** Code

## Theses (from outline §8)
- Code & data availability: repository (sim/phase_D/), frozen raw in *_raw*.json committed
  BEFORE analysis, full provenance by commit hashes (raw → analysis pairs per stage).
- Reproducibility: CPU-JAX float64, fixed PRNGKeys, pre-registration files (*_prereg.md +
  addenda) fixed before runs.
- AI methodology: Claude (architect/reviewer) designs + pre-registers; Claude Code
  (executor) implements + runs; the human owner holds decisions and veto. Pre-registration +
  kill-criteria + two-sided audit are controls over BOTH AI roles.

## Sources
- Repo tree: sim/phase_D/ (band4d, measurement, invariant, census, readout, run_*, analysis_*).
- Commit chain (per stage): "*-raw" (frozen) → "*-analysis"; canon "phase-D-canon*".
- Pre-registration set: D0/D1/D2/D2ext/S1runs/DS2/DS3 *_prereg.md (+ addenda).
