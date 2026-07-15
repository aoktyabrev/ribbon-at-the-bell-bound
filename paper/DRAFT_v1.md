<!-- СГЕНЕРИРОВАНО paper/build_draft.py — НЕ РЕДАКТИРОВАТЬ. Правки — в sections/*.md -->

# Full draft v1

Full draft for author review. Read from the repository file, not from any chat transcript.
Veto points marked: §2.1 [deferred-veto], §7.4 [author's position B], §7.6 [credo].

---
# Frontmatter

**Title:** The Ribbon at the Bell Bound: Mapping the Classical Boundary of a Geometric
Entanglement Ontology

**Author:** Artem Oktiabrev

**Affiliation:** Independent researcher, Ukraine

**Email:** aoktyabrev@gmail.com · **ORCID:** 0009-0003-3626-2002

**Note:** Spelling 'Oktiabrev' confirmed by author (passport form); contact email spelling
differs and is intentional.

---

# 1. Introduction

**Section owner:** Architect

Take seriously, for the length of one research program, the following
picture: an entangled pair is one extended object. Not two particles
sharing a state, but a single ribbon embedded in a dimension above the
three we see, its two ends being the two intersections with our slice —
so that the perfect correlations of the singlet would need no
communication, there being nothing to communicate between. The picture
has an immediate structural payoff (Section 2): the embedding carries a
ℤ₂ framing class, and from it the binary character of measurement
outcomes and the 720° spinor property descend without postulates.
The bet this program made — its maximal stake — was that the classical
dynamics of such an object, read by an honest basin measure, would
produce the quantitative statistics of the singlet: the isotropic
−cos θ correlation at unit amplitude, the Born weights, and a CHSH value
of 2√2. The registered expectation, written into the simulation
specification and in the record before the first physics campaign it
governs, was the opposite: "the default expectation, stated honestly
before launch: a sawtooth or p ≈ 1; p = 2 would be a major result
demanding paranoid verification; any outcome is informative" (SPEC §1,
translated from the Russian original; unedited since). The program aimed
at the quantum target while predicting, in writing, that it would
miss — and it missed exactly where the prediction said it would.

The bet was lost. This paper reports how it was lost — exactly,
quantitatively, and, we will argue, usefully. The classical ribbon
realizes more of the singlet than we find commonly appreciated:
anticorrelation of the right sign, a correlation amplitude that survives
chain length unchanged (a kinetic plateau, measured to N = 96 and
verified at two coupling stiffnesses), and a smooth cosine angular law —
though only along one privileged axis. What it cannot realize, we
can now name and number: exact outcome zeros are forbidden to it by a
structural theorem (the topological invariant that could enforce them
lives in precisely the fiber that axial measurement quotients out;
Section 4); honest isotropy collapses the cosine to the triangular
correlation of Bell's shared-randomness local model (Section 5); and
every apparent Bell violation the program produced — there were two —
died in a pre-registered audit, one to a false symmetry assumption, one
to a rediscovery of Pearle's detection loophole (Section 6). The
stiffness family the model spans approaches the Bell bound from below
and ends there: its limit point is CHSH = 2, not 2√2.

We consider the negative answer, so structured, to be the contribution.
A classical ontology was driven to a boundary; the boundary was mapped
rather than lamented; and the walls were named while the program was
still running — several of them named in advance in the pre-registration
record. The method that made this possible is plain: hypotheses with
kill-criteria committed before runs, raw data committed before analysis,
mirror controls, and a standing audit that treats any too-good result as
a defect until proven otherwise. That machinery caught not only the
physics artifacts above but the authors' own errors, in both directions
of the human–AI collaboration that executed the program (Section 8).

## 1.1 Related work and positioning

The walls this program hit are, of course, known walls. Bell's theorem
bounds every local hidden-variable account, and our isotropized
correlation is his shared-λ model met dynamically; the equivalence of
cosθ-linearity with no-signaling reproduces, as we found after the fact,
Gisin's 1990 argument — the uniqueness of the quadratic exponent within
the chord family being this program's own extension of it; the
post-selection artifact
of our second audit is Pearle's 1970 detection loophole; and the
impossibility of eliminating the Born postulate within the symmetric
pair is the constructive face of Kochen–Specker. We differ from the
classical hidden-variable literature — de Broglie–Bohm's nonlocal
dynamics, 't Hooft's cellular automata, Palmer's invariant-set theory —
not in outrunning these theorems, which we do not, but in genre: a
single ontology, pursued to its named limit, under pre-registration,
with every retreat documented and every rediscovery dated by commit.
To our knowledge the combination — an independently conceived geometric
ontology, a falsification-first simulation program, and a fully
version-controlled provenance trail — has not been reported in this
form. The reader who wants the theory is directed to Section 2; the
reader who wants the boundary, to Section 5; the reader who suspects
that classical models producing S > 2 must be hiding an error somewhere
will find two such errors, found and dissected, in Section 6.

---

# 2. Theory

**Section owner:** Joint (Architect + Author)

## 2.1 The ontological picture  [approved: architect, deferred-veto: author]

The ontological picture under test is this: an entangled pair is not two
particles bound by a mysterious connection, but a single extended object —
a ribbon — embedded in a space of dimension higher than the three we
observe. What we call the two particles are the two intersections of this
object with our three-dimensional slice; they are boundaries of one body,
not separate things. On this picture, the correlations of entanglement
would require no communication, because there is nothing to communicate
between — one object simply has one geometry. The picture earns its keep
at the level of structure: the embedding equips the ribbon with framing
classes labeled by ℤ₂ (since π₁(SO(3)) = ℤ₂), and from this single
topological fact two otherwise-postulated features of quantum measurement
descend for free — the binary character of outcomes, and the spinor
property that a 720° rotation, not 360°, returns the object to itself.
What this paper tests, to its limit, is whether the classical dynamics of
such an object can also produce the quantitative statistics of the singlet
state. It cannot — and the ways in which it cannot turn out to be sharply
nameable.

## 2.2 The chord law and the causal derivation of the exponent

The measurement model assigns outcome probabilities through a chord
measure: P(s,t | a,b) = |s·a − t·b|² / 8 for outcome signs s, t = ±1 at
clamp orientations a, b. [*] This law yields E = −cos θ, the half-angle
probabilities cos²(θ/2), no-signaling, and order-independence. [*] Its
quadratic exponent is not a choice. Embedding it in the one-parameter
family P ∝ |s·a − t·b|^p, any p ≠ 2 — and any admixture ε ≠ 0 of a
symmetric deformation — produces a superluminal telegraph on a partially
entangled link: the marginal at one end shifts with the remote setting
by Δ = 0.146 at p = 1, 0.081 at p = 3, 0.072 at ε = 0.2 (at q = 0.85,
Δ = ε·|(1−2q)³−(1−2q)| for the deformation family). [*] Causality pins
the Born exponent from both sides. As we later found, the core of this
argument — cosθ-linearity of the qubit correlation as an equivalent of
no-signaling — reproduces Gisin (1990); the two-sided uniqueness of
p = 2 within the chord family is this program's extension of it. [*]

## 2.3 The conservation law of the postulate

What causality can do, internal consistency cannot. A symmetric-pair
consistency battery — no-signaling, order-independence, mirror symmetry
— is passed by an entire ε-family of non-Born measures; consistency of
the pair does not fix the quadratic law. [*] Kochen–Specker (1967)
supplies the constructive complement: an outcome measure exists (the
honest cos²(θ/2) at a single end), but the postulate has then moved from
the probability rule into the measure over microconfigurations — it
relocates between rule, measure, and branch weights, and within the
symmetric pair it never annihilates. [*] Only the external principle of
causality (2.2) eliminates alternatives. We record this as the
conservation law of the postulate, and it shaped every later phase: any
mechanism that seemed to derive the Born weights was audited for where
it had hidden them instead. [*]

## 2.4 From analytical layer to dynamical program

Two analytical results sharpened into simulation targets. First, the
sawtooth: a uniform microphase measure with conical outcome basins —
the natural first reading of the ribbon — gives the triangular
correlation E = −ρ(1 − 2θ/π), the textbook local-realistic law, not the
cosine; this was recorded as an analytical counterexample in the
canonical log and became, unexpectedly, the exact destination of the
isotropized dynamical model (Section 5.4). [*] Second, basin measure
scaling as |chord|¹ gives the sawtooth while |chord|² gives Born — so
the question "where does the second power come from?" acquired a
geometric face and is posed, formalized but unsolved, in Section 7.3. [*]
The simulation program of Sections 3–5 is the systematic attempt to make
an elastic, thermal, topologically framed ribbon produce dynamically what
the analytical layer showed it must produce structurally — and the
measured ways in which it does not.

---

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

The analysis order was strict: raw measurements were frozen and committed before any fit,
flip, or comparison (each stage has a `*-raw` commit preceding its `*-analysis` commit). Every
campaign carried controls: an unbiased preparation sampler (boundary axes drawn uniformly on
S², sector preparation without bias, so the branch is never chosen in advance and the basin
decides; measurement.py), mirror pairs (a, b) ↔ (−a, −b), block convergence of the reported
observable, an explicit DEGENERATE class for undetermined basins, and — for the central
scaling claim — a pre-registered cross-seed audit of the quoted errors (Section 3.3). A
standing kill-criterion
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
(SPEC §5); all phase-D runs are CPU-JAX in float64. We chose float64 for the lift
bookkeeping: parity is the sign of a quaternion lift accumulated along the whole chain
(invariant.py), and we judged that a sign read off an accumulated product, together with its
conservation test (T-inv-3), wanted precision headroom we did not want to argue about. The
runs are CPU-only because fp64 on the available GPU is ~1/64 speed, and cells this size do
not need it.

Two distinct properties are worth separating. Repeatability is exact: every stage fixes a
PRNGKey seed protocol and commits its raw data, so the same seed returns the same number and
any figure can be regenerated from a named commit. Reproducibility — whether a result
survives a change of seed — is a separate question, and we measured it rather than assumed
it: a dedicated seed audit re-ran the N = 32 and N = 96 cells under twelve fresh base keys
and found the cross-seed scatter consistent with the quoted binomial errors (r = 0.88 and
1.11; Section 5.1, seed audit commit a26f76b).

---

# 4. No-go results

**Section owner:** Joint (Architect U(1)-theorem + Code D1 evidence)

The program's negative results are its most durable ones. We state them in
increasing order of strength: a gauge-blindness property of axial
observables, its sharpening into a structural theorem about where the
topological invariant lives, and the census result showing that topology
constrains no outcome by measure alone.

## 4.1 Gauge blindness of axial observables

Every observable used by a physical measurement in this model is axial: a
function of the boundary axis n = R(q)ê, where R is the rotation obtained
from the frame quaternion q under the double cover SU(2) → SO(3). Any
such function factors through SO(3) by construction, and the kernel of the
covering map is exactly the ℤ₂ we care about: two frame configurations
differing only by the lift sign (q vs −q) produce identical axial fields
everywhere, hence identical statistics for every axial observable. In
phase B/C this appeared as an empirical wall — kinks without topological
protection relaxed away, and charged configurations, with energies of order
k_e·(π/2..π)², remained thermally unpopulated for T ≪ k_e. At that stage
one could still hope that a cleverer local readout might see the charge.

## 4.2 The U(1) theorem: axis ≠ frame

Phase D closed that hope. In the R⁴ embedding, the boundary clamp fixes an
axis, but an axis determines a frame only up to a residual U(1) of rotations
about it — and the ℤ₂ framing class lives precisely in that residual layer,
as the boundary sign of the lift. The construction is explicit: the
lift-twin test builds pairs of configurations with identical axial fields
n_i at every site (max |Δn| = 0) and opposite parity. Every candidate
readout we constructed — coorientation of the frame against the slice
normal, the sign of the nearest interior w-coordinate, windowed relative
lift — passed the locality test (stable at window k ≤ 8) and, without
exception, returned identical statistics on the twins. The blindness was
not the price of enforcing locality; every local slice datum we could build
was blind. We registered this in advance as a named wall ("blind slice"),
and it fired. The theorem-level statement: a ℤ₂ framing invariant of an
embedded curve in R⁴ is invisible to any boundary observable that factors
through the axis, because ℤ₂ ⊂ ker(S³ → SO(3)) — the invariant is stored
exactly in the fiber that axial physics quotients out. Binarity and
spinority thus descend from the framing topology (Section 2) while
remaining unreadable by the very measurements whose outcomes they were
meant to explain. This tension is, we believe, the sharpest single lesson
of the program.

## 4.3 Sector blindness of the axial signal

The complement of 4.2, measured directly: at matched preparation, the axial
correlation amplitude is statistically identical in the even and odd
topological sectors (ΔA = −0.040 ± 0.038 at N = 32, M = 1200). The
correlation carried by axial readout is generic chain correlation,
controlled by coupling stiffness (Section 5), not by topology. An earlier
apparent sector shift (~0.07 at M = 200) did not survive the statistics
increase and is withdrawn.

## 4.4 The census result: topology forbids no outcome

Pre-registered hypothesis D-H3a proposed that in the odd sector at θ = 0
the aligned outcome branch has empty support — a geometric superselection
that would have supplied the exact zeros the Born rule requires. The
census refuted it: all 16 cells of {sector} × {branch} × {θ ∈ {0, π}}
contain non-singular representatives, found both numerically (constrained
minimization from ≥20 diverse starts per cell) and by an explicit
homotopy argument — the residual U(1) layer of 4.2 absorbs the framing
constraint, so a 2π twist of the fiber connects nominally forbidden
configurations to allowed ones. The kill-criterion fired as
registered: there is no measure-zero prohibition in this geometry.
Combined with 4.2, the two results close both readings of the phase-D
bet: topology cannot be read locally, and it forbids nothing globally.
What survives of it is one number — the scale-invariant amplitude of
Section 5 — and the structural explanation of binarity itself.

---

# 5. Quantitative boundary

**Section owner:** Code

## 5.1 The plateau

The one quantity the ribbon carries that does not dilute with length is the axial readout
amplitude of the odd sector. Fit against three models — a constant A∞, a power law A·N^−γ,
and a saturating A∞ + c·N^−γ — the constant wins decisively: A∞ = 0.363 ± 0.012, with the
power law disfavoured by ΔAICc = 6.36 and its exponent fitted negative (no decay) (D2-ext;
commit 2784edf). The plateau holds at a second stiffness: cross-scanning N ∈ {16, 32, 64, 96}
at k_f×1 and k_f×4, the amplitude change from N = 16 to N = 96 is +0.010 ± 0.038 and
+0.002 ± 0.023 respectively — flat within error at both couplings (DS2; commit f928dd4). The evidence is
bounded: N ∈ [16, 96], k_f ∈ {×1, ×4}. Within that window the plateau is kinetic, not
thermodynamic — the amplitude is flat in temperature (A(T) = 0.368 / 0.397 / 0.353 at
T = 0.025 / 0.05 / 0.10, unchanged within σ ≈ 0.027) (S1-runs; commit a9cef7b) — so it is a
property of the basin structure, not of thermal fluctuation. The quoted errors are binomial,
and a dedicated seed audit confirms they are honest as a measure of reproducibility: eleven
independent repeats of the N = 32 cell scatter with s_seed = 0.024 against a binomial
σ = 0.027, a ratio r = 0.88 (χ² = 7.7 / 10, p = 0.66), and the ratio does not grow with chain
length (r = 1.11 at N = 96) (seed audit; commit a26f76b).

[Fig. 1: plateau — d2ext_scaling.png + ds2_cross.png]

## 5.2 Origin of the amplitude

The amplitude is stiffness-controlled. Sweeping the twist coupling gives
A(k_f) = 0.27 / 0.42 / 0.56 / 0.84 for k_f × {0.5, 1, 2, 4} — a strong
monotone dependence that refutes any fixed geometric-measure origin
(S1-runs R1; commit a9cef7b). The plateau value A∞ = 0.363 ± 0.012 belongs
to the D2-ext campaign (N ∈ [16, 96] at its baseline setup); the stiffness
sweep, run at N = 32, gives 0.418 ± 0.026 at nominal k_f×1 — consistent in
scale, not identical — the gap reflects seed-to-seed sampling scatter,
quantified in the seed audit (s_seed = 0.024). This only sharpens the point: the amplitude is the
reading of a dial, not a constant of the model; the weakness of the signal
is a property of soft coupling, not a limit of the mechanism. At θ = 0 the
amplitude is, by identity, A = 2·P_aligned − 1 (P_aligned = 0.681 for the
D2-ext plateau), so explaining the amplitude means explaining the
aligned-basin weight. Two measured facts locate that
weight in kinetics rather than thermodynamics. First, it is flat in
temperature (Section 5.1), while any Boltzmann reading of the landscape
would move with T: the census finds the four branch minima spread over
0.22 in energy (branch means −3.78 for both branches, Δ ≈ 0.006), and no
assignment of Boltzmann weights to these energies reproduces a
temperature-independent 68/32 split (D1 census; D_synthesis_S1 §2).
Second, it is flat in chain length from N = 16 to N = 96 at both measured
stiffnesses (Section 5.1), while an equilibrium chain correlation with a
finite correlation length would decay. What remains is relaxation itself:
the split is set by the geometry of the attraction domains under the
clamped dynamics — a domain volume controlled by coupling stiffness. This
is the reading fixed in the canonical record after the cross-scan (commit
311d3ae): the amplitude is a stiffness-controlled kinetic correlation,
athermal and sector-blind (Section 4.3).

![Fig. 2: A∞(k_f) — stiffness memory](../../sim/phase_D/fig/s1runs_kf.png)

*Fig. 2. A∞(k_f) — stiffness memory. Four points with 1σ error bars from the frozen
S1-runs R1 raw data (N = 32, M = 1200, T = 0.05; commit a9cef7b). Dotted line: the
D2-ext plateau A∞ = 0.363 of Section 5.1, measured over N ∈ [16, 96].*

## 5.3 Anisotropy map

The cosine angular law the correlation appears to follow is anisotropic. Tilting the clamp axis
a by α from the privileged axis ê and reading the antiparallel amplitude A(α) = |E_anti|, the
signal decays smoothly to zero at α = π/2: from 0.387 to 0.005 at k_f×1 and from 0.865 to 0.008
at k_f×4 (DS3; commit 0fb5452). The best-fit law is cos α at k_f×1 and steepens to ~cos²α (an
exponential fitting equally well) at k_f×4. The axis ê — simultaneously the twist axis and the
readout axis — is strongly privileged; the cosine dependence exists only along it, which is
exactly why the isotropic CHSH estimator of Section 6 was invalid.

[Fig. 3: anisotropy map — ds3_aniso.png]

## 5.4 The trilemma and the triangle

Averaged honestly over the shared randomness λ = (R, n_A, n_B) — where R ~ Haar on SO(3) is
the ribbon's orientation, common to both ends through the geometry, while the settings n_A and
n_B are drawn independently — the correlation is not a cosine. Fitting the
isotropized E(θ) against both forms, the triangular local-realistic function E = −ρ(1 − 2θ/π)
beats the cosine by a wide margin in χ²: 8 vs 39 at k_f×1 and 1 vs 218 at k_f×4 — the data lie
on the straight line, not the curve (DS3; commit 0fb5452). This is the shared-λ local model's
signature, and its CHSH value is exactly S = 2ρ, with ρ the source-alignment amplitude:
ρ = 0.374 / 0.810 gives |S| = 0.75 / 1.62 at k_f × {1, 4}. Three properties therefore trade
off and cannot be held together: a cosine form exists only anisotropically (5.3); honest
isotropy forces the triangle (this subsection); and the amplitude ρ is bought with stiffness
(5.2). The family, swept in k_f, approaches the Bell bound from below — |S| = 1.62 at k_f×4
sits a measured distance 2 − 1.62 = 0.38 short of it — with the deterministic axial limit
(ρ → 1, |S| → 2) as its endpoint.

[Fig. 4: triangle vs cosine — ds3_iso.png]

## 5.5 Synthesis

The k_f family is a classical frontier: a one-parameter sweep of local models whose limit
point is the Bell bound (CHSH → 2), reached from inside the classical region. The quantum
target — an isotropic cosine with unit amplitude and CHSH = 2√2 — lies outside the family on
three independent axes at once: amplitude (bounded by ρ < 1), form (isotropy forces a triangle,
not a cosine), and isotropy itself (the cosine survives only along the privileged axis). No
single knob moves the ribbon toward it; that is the quantitative content of the phase-D bet's
outcome, and its interpretation is deferred to Section 7.

---

# 6. Case studies in self-correction

**Section owner:** Joint (Code audits + Architect methodology)

A hidden-variable research program faces a specific epistemic hazard: the
researcher wants the model to violate a Bell inequality, and the pipeline is
built by the same people who hold that hope. We therefore pre-registered,
before any CHSH campaign, a standing kill-criterion: any result S > 2 in a
manifestly local model is treated as evidence of a protocol error, triggering
a full audit before any interpretation. This criterion fired twice. Both
times it caught an artifact of method, not physics — and both artifacts turn
out to be textbook failure modes with decades of history in the Bell-test
literature. We report them in detail because we believe the mechanism of
their capture is a result in its own right.

## 6.1 Audit 1: the false-isotropy artifact (DS2)

Applying the standard isotropic CHSH estimator to our stiff-coupling data
(k_f×4) yielded |S| = 2.58, comfortably above the classical bound. The
pre-registered trigger halted interpretation. Direct measurement of all four
correlators — rotating both boundary clamps rather than assuming rotational
symmetry — revealed E(π/2, π/4) ≈ 0.007 against E(0, π/4) = −0.595: the
system is strongly anisotropic, with the twist axis privileged, and the
isotropic estimator is simply invalid for it. The directly measured value is
|S| = 1.21, safely sub-classical. The audit had a retroactive cost we accept
openly: all previously quoted CHSH values (phase C: 1.48; D2: 0.73–1.25) were
computed with the same invalid estimator and are withdrawn. They remain in
the canonical record, struck through, with the revision documented (commit
311d3ae).

## 6.2 Audit 2: the detection loophole, rediscovered from the inside (DS3)

The isotropization protocol (a source axis drawn Haar-uniformly per replica —
honest shared randomness, available to both ends through the geometry of the
ribbon but not to the experimenter) produced |S| = 2.39. Since the
construction is manifestly local, Bell's theorem guarantees this is a bug;
the trigger fired again. The audit located it in event selection: replicas
with a weak axial projection (|proj| < 0.2) were being discarded as
DEGENERATE — and the discard rate itself varied with the measurement
setting, from 21% of events at the correlation extremes (θ = 0, π) to a peak of
37–38% at intermediate angles (36–37% at the CHSH angles π/4 and 3π/4). This setting-dependence is
not incidental to the artifact; it is the artifact. Setting-dependent post-selection is precisely the
detection loophole of Pearle (1970) — the class of local models that forced
four decades of loophole-closing in experimental Bell tests. Our pipeline
reproduced the disease and, once the cut was removed (all events kept,
sign(0) → +1 by convention), the cure: |S| = 1.62 at k_f×4, again
sub-classical.

*Table 1. Every CHSH value the program produced, and its fate. The isotropic estimator
S = 3E(π/4) − E(3π/4) assumes E(a, b) = E(|a − b|); the ribbon is strongly anisotropic
(Section 5.3), so every value computed with it is withdrawn. Withdrawn values are struck
through, not deleted (canon "Ревизия CHSH"; commit 311d3ae).*

| value | campaign | method | status | where corrected |
|---|---|---|---|---|
| ~~≈ 1.48~~ | phase C | isotropic estimator | withdrawn: invalid isotropic estimator | superseded by DS2 direct (§6.1) |
| ~~0.73–1.25~~ | D2 | isotropic estimator | withdrawn: invalid isotropic estimator | superseded by DS2 direct (§6.1) |
| ~~2.39~~ | DS3 (primary) | direct, but with setting-dependent DEGENERATE post-selection (\|proj\| < 0.2, ~36% of replicas) | withdrawn: setting-dependent post-selection — the detection loophole (canon also charges the isotropic estimator: double artifact) | `analysis_ds3`, post-selection removed → 1.62 (§6.2; commit 0fb5452) |
| **1.21** | DS2 | direct, both ends rotated, no post-selection | valid: direct | — |
| **0.75 / 1.62** | DS3 | isotropized shared randomness λ = (R, n_A, n_B); CHSH = 2ρ at k_f × {1, 4} | valid: direct | — |

No value survives audit above 2; the ribbon is always sub-classical.

## 6.3 What the two audits establish

First, a negative result of unusual strength: across the entire program, no
Bell violation survived an audit. Every S > 2 traced to an identifiable
protocol error, and the corrected values sit strictly inside the classical
region — converging, in the stiff-coupling limit, toward the Bell bound from
below (Section 5). Second, a methodological claim: the pre-registered
"too-good-to-be-true" trigger is not decoration. Both artifacts — invalid
symmetry assumption, setting-dependent post-selection — are historically
documented failure modes that took the community years to identify in
experimental contexts. A standing kill-criterion, registered before the data
existed and owned by no one at the table, identified both within a single
analysis cycle each. Third, a note on provenance: a theoretical
entry in our canonical log (the "Bell trap", §2.7 of the model note) derives
that any shared-λ, locally deterministic readout of the ribbon must produce
the triangular correlation function — a forced consequence, not a guess; the
isotropization campaign landed on exactly that triangle (Section 5). We
deliberately rest this point on the logical status of the prediction rather
than on its calendar priority, which our commit history cannot independently
establish. The program's negative
predictions were confirmed by its own later data — which is, we would argue,
the behavior one should demand of an honest hidden-variable program before
trusting any of its positive claims.

---

# 7. Discussion

**Section owner:** Architect

## 7.1 The final lemma

Phase D leaves the program with a lemma sharper than the one it started
with. The classical ribbon realizes, from geometry alone: the binary
character of outcomes and their spinor structure (from the ℤ₂ framing
class), anticorrelation (the singlet sign), a correlation amplitude
that does not dilute with chain length (kinetic plateau, two stiffnesses,
N ≤ 96), and a smooth cosine angular law — but only along a privileged
axis. It does not realize: exact zeros (no outcome is forbidden by
measure, and the invariant that could forbid one is unreadable by axial
measurement — Sections 4.4 and 4.2), an isotropic cosine (honest orientation averaging
forces the triangular law; Section 5.4), or any Bell violation (every
S > 2 died in audit; Section 6). The boundary between the two lists is
not a fog but a set of named, quantified walls.

## 7.2 Relation to the classical theorems

The program repeatedly rediscovered, from the inside and with its own
numbers, results that mark the classical-quantum boundary. The equivalence
of cosθ-linearity (the qubit Born rule) with the no-signaling constraint
reproduces Gisin (1990); the uniqueness of the Born exponent p = 2 within
the chord family is this program's own extension of the same causal
argument; the triangular correlation of the isotropized ribbon is Bell's
shared-λ local model, a textbook construction that the program first met
as an analytical counterexample (the "saw", session 3) and then found
realized dynamically by its own elastic ribbon — the analytical
counterexample and the dynamical outcome bracketing the program; the
setting-dependent post-selection artifact of DS3 is Pearle's (1970)
detection loophole, reproduced and repaired in one audit cycle; and the
relocation of the Born postulate between rule, measure, and branch
weights — with no elimination available within the symmetric pair itself,
only by the external principle of causality (Section 2) — is the
constructive face of Kochen–Specker (1967). The number makes the wall
tangible twice over: even the family's deterministic limit point (ρ → 1)
yields the triangular E(45°) = −0.500 — the value the session-3 analytical
construction already produced (−0.502, canon §2.7) — against the quantum
−0.707; and the measured ribbon does not reach even that, giving −0.187 and
−0.405 at the two stiffnesses. We take the convergence itself as evidence: a program that
honestly explores a classical ontology does not wander freely — it is
funneled onto the same walls the theorems describe, and arrives at them
with measured coordinates.

## 7.3 The open problem

One exit remains formally open, registered in the canonical record as the
open problem E: whether a physical fiber-breaking invariant of the R⁴
embedding (a second-fundamental-form datum, or a preferred slice normal)
can produce an empty aligned-branch basin together with a smooth isotropic
cosine — that is, an escape from the trilemma of Section 5.5. We
record the obstacles as honestly as the opening — both are our judgment
from the phase-D results, not established results themselves: any energetic anchoring
of the frame is another stiffness dial, moving the model along the
classical family rather than off it; and a shared external frame is
shared randomness, bounded by Bell's theorem regardless of its geometric
pedigree. A related conjecture from the theory line — that basin measure
scaling as the square of the chord ("the ribbon has a face") would
reproduce the Born weights — is formalized to the point of an explicit
measure, μ(B(s,t)) = |s·a − t·b|²/8, and posed as an open problem —
construct it for all clamp orientations, or prove an impossibility lemma
naming the minimal missing structure; it is unsolved, not untested
folklore [conjecture]. We consider these open in the technical sense, not
the hopeful one.

## 7.4 The status of the ontology, and of the classical route

The ontological picture itself — the pair as one extended object, with
binarity and spinority descending from its framing topology — is not
refuted by anything in this paper. What is refuted, by construction and
at scale, is the proposition that the *classical dynamics* of such an
object reproduces singlet statistics. We state our position beyond the
minimal one: we regard the classical route for this ontology as
exhausted. The U(1) theorem shows that the topological invariant carrying
the ontology's explanatory power is structurally unreadable by axial
measurement; the trilemma shows that amplitude, angular form, and
isotropy cannot be purchased together anywhere in the accessible family;
and the family's limit point is the Bell bound, approached from below.
Further classical modifications of this model, other than a solution to
the open problem E, would in our judgment rediscover these walls under
new names. If the ribbon picture has a future, it lies in a quantum
dynamics of the extended object — a question this paper does not open.

## 7.5 Limitations

The scope of the evidence is narrower than the scope of the claims a
reader might extrapolate, and we state it exactly. The dynamics tested is
classical throughout; nothing here bears on a quantum dynamics of the
same object. The discretization is one class among the possible: a framed
curve in R⁴ with axial boundary readout — a triangulated surface, or
non-axial boundary couplings, were designed around but not simulated. The
U(1) theorem is a statement about axial (SO(3)-factorizable) boundary
data, not about all conceivable measurements. And the quantitative
results are bounded: N ∈ [16, 96], k_f ∈ {×1, ×4}, one temperature regime
verified flat. Within these bounds the walls are measured; beyond them
they are extrapolated.

## 7.6 What we take to be the product

Beyond the specific results, we submit the method as a product in its own
right: an ontology driven to its named limit by pre-registration,
kill-criteria owned by no one at the table, raw-before-analysis commit
discipline, and a standing audit that fires on results that are too good.
This machinery caught two textbook artifacts within one analysis cycle
each (Section 6) and forced every retreat to be documented rather than
forgotten. The program's answer to its opening bet is negative, exact,
and reusable — and the same machinery is now free to be pointed at the
next idea. Everything should be tested; that is how one either learns
something new, or confirms something known — and both are wins.

---

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
`*_prereg.md`, and what the record proves is graded (Section 3.2): no pre-registration was
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

---


---

## Figure and table list

All figures are produced by committed scripts from committed raw data; each can be
regenerated from a named commit (Section 8.1).

| # | subject | file | script | section |
|---|---|---|---|---|
| Fig. 1 | plateau A(N) | `d2ext_scaling.png`, `ds2_cross.png` | `analysis_ext.py`, `analysis_ds2.py` | 5.1 |
| Fig. 2 | A∞(k_f), stiffness memory | `s1runs_kf.png` | `plot_s1runs_kf.py` | 5.2 |
| Fig. 3 | anisotropy map A(α) | `ds3_aniso.png` | `analysis_ds3.py` | 5.3 |
| Fig. 4 | triangle vs cosine (isotropized) | `ds3_iso.png` | `analysis_ds3.py` | 5.4 |
| Table 1 | CHSH revision: withdrawn vs valid | inline | — | 6.2 |

Figure files live in `sim/phase_D/fig/`.
