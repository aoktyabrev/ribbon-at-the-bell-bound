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
