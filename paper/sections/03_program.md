# 3. Simulation program

**Section owner:** Code

## 3.1 Four phases

Phase A built the computational scaffold — pure JAX (jit, vmap, lax.scan; no Python loops
over steps or batch), float32 state as explicit pytrees — and passed the R0 null
control, the baseline that later grew into the regression suite carried through phase C
(commit 86e2154, "phase-A: skeleton+tests+R0 green"). It fixed no physics on its own; it
fixed the ground on which every later claim would have to stand.

Phase B established gauge blindness. Any observable that is a function of the boundary axes
n = R(q)ê factors through the covering map SU(2) → SO(3) and is therefore identically blind
to the ℤ₂ lift; this is a theorem (Section 4), first met here as an empirical wall. The
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
estimator we later withdrew; see Section 6.)

Phase D embedded the framed ribbon in R⁴ ≡ ℍ, turning the integer twist (ℤ) into a ℤ₂ framing
class (π₁(SO(3)) = ℤ₂). Across the D0–DS3 sequence — infrastructure and invariant (D0), the
readout operator and empty-sector test (D1), scaling (D2/D2-ext), amplitude origin (S1),
cross-scan and CHSH audits (DS2/DS3) — the single number that survived is the scale-invariant
readout amplitude A∞ = 0.363 ± 0.012 (D2-ext; commit 2784edf), whose meaning and limits are
the subject of Section 5.

## 3.2 Methodology

Every experiment was pre-registered before it was run. A registration fixed a hypothesis, a
quantitative expectation, and a kill-criterion in a committed `*_prereg.md`, not edited
afterward. A representative entry (D0, hypothesis D-H2; translated from the Russian original)
reads: "ℤ₂ parity is exactly conserved
on accepted trajectories, and the singular-rejection fraction → 0 as dt → 0; kill: if the
rejection fraction does not fall with dt, the parity is an artifact of the filter"
(D0_prereg.md). Registering the failure condition in advance is what let a null result count
as a result rather than as a disappointment to be explained away.

The analysis order was strict: raw measurements were frozen and committed before any fit,
flip, or comparison (each stage has a `*-raw` commit preceding its `*-analysis` commit). Every
campaign carried controls: an unbiased preparation sampler (boundary axes drawn uniformly on
S², sector preparation without bias, so the branch is never chosen in advance and the basin
decides; measurement.py), mirror pairs (a, b) ↔ (−a, −b), block convergence of the reported
observable, and an explicit DEGENERATE class for undetermined basins. A standing kill-criterion
treated any CHSH > 2 in a manifestly local model as a protocol error to be audited before
interpretation; it fired twice, and both times caught a method artifact rather than physics
(Section 6).

This machinery was not incidental to the working arrangement but demanded by it. The program
was run by an AI architect (design and pre-registration), an AI executor (implementation and
runs), and a human owner (decisions and veto) (Section 8). Pre-registration, kill-criteria,
and the raw-before-analysis order are controls over both AI roles: they bind the design and
the execution to commitments made before the data existed, which is precisely the point at
which the shared hope for a positive result would otherwise leak into the pipeline.

## 3.3 Infrastructure

Phase D reused the phase A–C toolkit directly: the quaternion frame algebra, energy, and
Langevin dynamics carried over into the R⁴ model (band4d.py, measurement.py). Tests were
organized by INVARIANT rather than by count: Bishop SO(4) transport (R·t = t′, RᵀR = I,
det = +1, identity on the orthogonal complement — minimality), the ℤ₂ parity invariant
(T-inv-1/2/3: straight rod → +1, inserted 2π → −1, 4π → +1, and invariance under 1000 smooth
deformations), mirror symmetry, and singularity detection (temporal lift jump, tangent
reversal, spatial lift-wall). The phase-D suite (9 tests) and the phase A–C suite (60 tests)
both run green under this coverage. Phases A–C run in float32 per the simulation defaults
(SPEC §5); all phase-D runs are CPU-JAX in float64, required by the lift bookkeeping: parity
lives in the sign of an accumulated quaternion lift, and its conservation test (T-inv-3)
demands precision headroom that float32 does not guarantee (invariant.py, conftest.py). The
runs are CPU-only because fp64 on the available GPU is ~1/64 speed, and cells this size do
not need it. Every stage uses a fixed PRNGKey seed protocol and commits its raw data,
so any figure can be regenerated from a named commit.
