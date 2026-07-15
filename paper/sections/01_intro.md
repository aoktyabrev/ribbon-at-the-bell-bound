# 1. Introduction

**Section owner:** Architect

Take seriously, for the length of one research program, the following
picture: an entangled pair is one extended object. Not two particles
sharing a state, but a single ribbon embedded in a dimension above the
three we see, its two ends being the two intersections with our slice —
so that the perfect correlations of the singlet would need no
communication, there being nothing to communicate between. The picture
has an immediate structural payoff (Section 3): the embedding carries a
ℤ₂ framing class — the framing topology supplies a natural ℤ₂-valued
structural label and the homotopy underlying the 4π belt-trick return.
A physical map from this label to measurement outcomes is an additional
requirement — and Section 5 proves that for axial readout no such map
exists.
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
can now name and number: the topology does not enforce exact outcome
zeros — the census found no topologically forbidden branch, and the
framing invariant is inaccessible to axial readout (a structural theorem:
the invariant that could enforce them lives in precisely the fiber that
axial measurement quotients out; Section 5); honest isotropy collapses the
cosine to the triangular
correlation of Bell's shared-randomness local model (Section 6); and
every apparent Bell violation the program produced — there were two —
died in a pre-registered audit, one to a false symmetry assumption, one
to a rediscovery of Pearle's detection loophole (Section 7). The
measured family moves toward the Bell bound as stiffness grows
(|S| = 1.62 at k_f×4); its deterministic axial limit is expected — not
measured — to reach ρ → 1, S → 2, not 2√2.

We consider the negative answer, so structured, to be the contribution.
A classical ontology was driven to a boundary; the boundary was mapped
rather than lamented; and the walls were named while the program was
still running — several of them named in advance in the pre-registration
record. The method that made this possible is plain: hypotheses with
kill-criteria committed before runs, raw data committed before analysis,
mirror controls, and a standing audit that treats any too-good result as
a defect until proven otherwise. That machinery caught not only the
physics artifacts above but the authors' own errors, in both directions
of the human–AI collaboration that executed the program (Section 9).

## 1.1 Related work and positioning

The walls this program hit are, of course, known walls. Bell's theorem
[Bell1964] bounds every local hidden-variable account, and our isotropized
correlation is his shared-λ model met dynamically; our causal argument for
cosθ-linearity is, as we found after the fact, the measurement-rule
analogue of Gisin's no-go for nonlinear quantum dynamics [Gisin1989;
Gisin1990] — there, nonlinearity of the evolution enables superluminal
signaling; here, nonlinearity of the outcome rule in cos θ does, the
argument's form being Gisin's and its object ours — with the uniqueness of
the quadratic exponent within the chord family this program's own
extension; the post-selection artifact
of our second audit is Pearle's detection loophole [Pearle1970]; and the
impossibility of eliminating the Born postulate within the symmetric pair
is our own methodological observation, standing in the historical line of
Kochen–Specker's demonstration that an outcome measure exists [KS1967] rather
than following from that theorem. We differ from the
classical hidden-variable literature — de Broglie–Bohm's nonlocal dynamics
[Bohm1952a; Bohm1952b], 't Hooft's cellular automata [tHooft2016], Palmer's
invariant-set theory [Palmer2009] —
not in outrunning these theorems, which we do not, but in genre: a
single ontology, pursued to its named limit, under pre-registration,
with every retreat documented in a version-controlled record — and every
rediscovery made after that record began, dated by commit.
To our knowledge the combination — an independently conceived geometric
ontology, a falsification-first simulation program, and a fully
version-controlled provenance trail — has not been reported in this
form. The reader who wants the theory is directed to Section 3; the
reader who wants the boundary, to Section 6; the reader who suspects
that classical models producing S > 2 must be hiding an error somewhere
will find two such errors, found and dissected, in Section 7.
