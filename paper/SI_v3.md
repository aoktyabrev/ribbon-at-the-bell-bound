# Supplementary: program history and infrastructure

*Supplementary to Section 5 (Methods) of the main text. Moved here per the structural
revision: this is phase chronology and implementation detail, not method.*

## SI-P.1 The four phases

Phase A built the computational scaffold — pure JAX (jit, vmap, lax.scan; no Python loops
over steps or batch), float32 state as explicit pytrees — and passed the R0 null
control, the baseline that later grew into the regression suite carried through phase C
(commit 86e2154, "phase-A: skeleton+tests+R0 green"). It fixed no physics on its own; it
fixed the ground on which every later claim would have to stand.

Phase B established gauge blindness. Any observable that is a function of the boundary axes
n = R(q)ê factors through the covering map SU(2) → SO(3) and is therefore identically blind
to the ℤ₂ lift; this is a theorem (Section 5), first met here as an empirical wall. The
operative number was energetic: kinks carry energy of order k_e·(π/2..π)² and remain
thermally unpopulated for T ≪ k_e, so the charged sector never populates spontaneously
(ribbon_model_note.md § "Дополнение (июль 2026, после фазы B симуляции): no-go
гейдж-слепоты и роль вложения в 4D").

Phase C produced the entropic singlet. With a conserved ℤ₂ charge (the 2π twist sector) read
under an entropic basin measure, the correlation takes the singlet form
E(θ | 2π) = −tanh(ΔS·cosθ / 2) with ΔS ∝ cosθ (ribbon_model_note.md § "Резолюция открытой
задачи §3 (фаза C симуляции, июль 2026)"; phase_C_report). The final phase-C lemma fixed
its ceiling: the amplitude tanh(ΔS/2) ≈ 0.5
stays strictly below 1, so exact zeros — and with them the Born rule — are unreachable by a
classical entropic measure. The ceiling was double: the phase-C lemma also recorded that the
effect was finite-size — the 2π charge is carried by a twist soliton, localized (peak twist
density ≈ 0.9), whose effect passes through a crossover in chain length — ferromagnetic sign
up to N−1 ≈ 33, an antiferromagnetic window near 48–64, extinction by N−1 ≈ 95 — so the
singlet form is finite-size and fades as N → ∞ (phase_C_report §7). Exact zeros and
scale invariance are both defining properties of the quantum correlation; the classical
entropic mechanism had neither. These two missing invariances are precisely the questions
phase D inherited. (The CHSH figure quoted at the time was computed with an
estimator we later withdrew; see Section 7.)

Phase D embedded the framed ribbon in R⁴ ≡ ℍ, turning the integer twist (ℤ) into a ℤ₂ framing
class (π₁(SO(3)) = ℤ₂). Across the D0–DS3 sequence — infrastructure and invariant (D0), the
readout operator and empty-sector test (D1), scaling (D2/D2-ext), amplitude origin (S1),
cross-scan and CHSH audits (DS2/DS3) — the single number that survived is the length-stable
readout amplitude (no resolved decay over the measured range) A_plateau = 0.363 ± 0.012 (D2-ext; commit 2784edf), whose meaning and limits are
the subject of Section 6.

## SI-P.2 Infrastructure and test coverage

Phase D reused the phase A–C toolkit directly: the quaternion frame algebra, energy, and
Langevin dynamics carried over into the R⁴ model (band4d.py, measurement.py). Tests were
organized by INVARIANT rather than by count: Bishop SO(4) transport (R·t = t′, RᵀR = I,
det = +1, identity on the orthogonal complement — minimality), the ℤ₂ parity invariant
(T-inv-1/2/3: straight rod → +1, inserted 2π → −1, 4π → +1, and invariance under 1000 smooth
deformations), mirror symmetry, and singularity detection (temporal lift jump, tangent
reversal, spatial lift-wall). The phase-D suite (9 tests) and the phase A–C suite (60 tests)
both run green under this coverage. Phases A–C run in float32 per the simulation defaults
(SPEC §5); all phase-D runs are CPU-JAX in float64. We chose float64 for the lift
bookkeeping: parity is the sign of a quaternion lift accumulated along the whole chain
(invariant.py), and we judged that a sign read off an accumulated product, together with its
conservation test (T-inv-3), wanted precision headroom we did not want to argue about. The
runs are CPU-only because fp64 on the available GPU is ~1/64 speed, and cells this size do
not need it.

---

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

---

