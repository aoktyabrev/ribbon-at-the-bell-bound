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

[Table 1: CHSH revision — retracted vs valid values]

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
