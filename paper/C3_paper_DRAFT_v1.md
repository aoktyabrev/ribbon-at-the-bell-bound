# Born from causality, Tsirelson from steering, and the amplitude seam:
# an internal reconstruction inside a geometric hidden-connection model

**Cycle 3 draft v1.** Author: A. Oktyabrev (aoktyabrev@gmail.com).
Status: assembly draft (~11 pp + battery appendix). All numbers cite raw
as `file (commit)`. Claim-discipline per program standard: theorem only with
written proof + battery + review; conjecture otherwise.

---

## Abstract
Within a geometric hidden-connection ("ribbon") model we close, and then
delimit, an internal reconstruction of quantum correlations. (i) Where a
mechanism carries **steering** — conditional states with internally
generated diversity D(χ)>0 — the no-signaling prohibition forces the Born
exponent p=2 uniquely; the derivation is internal (the steering ensembles
are produced by a shifted source measure, not imported), executable, and
robust under five deformations of the measure. (ii) Mechanisms that resolve
outcomes **locally** are capped by Bell's theorem at S=2, with a **1/3-cosine
point** (E=cosθ/3) as their isotropic-cosine ceiling, reached independently
three ways. (iii) The **amplitude seam** between that ceiling and the quantum
value S=2√2 is not crossed by decorating local mechanisms — three registered
attempts fail on a trilemma of horns (no steering / no amplitude / superluminal
telegraph) — but only by **joint, nonlocal resolution**, which by our
foliation results demands a preferred frame that the statistics render
invisible. The bridge statement is an exact division of labor, not a
unification.

---

## 1. Introduction — the import and the question
The program's canonical text acknowledged an **import**: partially entangled
ensembles were introduced by hand ("within this ansatz… we make no claim of
a general derivation of the Born rule", FINAL_v1:284–290). Cycle 3 asks
whether the steering that underlies the Born rule can be **generated
internally** by the model's own geometry, and — once it is — what exactly
stands between such a mechanism and full quantum correlations. The answer
assembles into four registered statements (RULE, CEILING, SEAM, FRAME),
each with kill-criteria and public commits.

## 2. Model
**Source / steering (F2).** A shared connection orientation λ on S² is drawn
from a shifted measure μ_χ ∝ 1+χλ_z. A measurement setting a partitions λ by
s=sign(λ·a); the conditional ensemble's **raw** (sub-unit) mean vector m_s is
the steered state; readout probability f_p(c)=|1+c|^{p/2}/(|1+c|^{p/2}+|1−c|^{p/2}),
with f_∞ the step 1{c>0}. **Lesson (record):** normalizing the conditional
axis broke total probability and produced Δ(p=2)>0 — the "Δ(2)>0 = bug"
criterion caught it; the raw sub-unit Bloch vector is correct
(battery_c3b, 968aaa5).

**Joint / chord layer.** A postulated pair law P(s,t|a,b)=(1−st cosθ)/4
(E=−cosθ, S_max=2√2), and its weighted generalization |s√w_a a − t√w_b b|^p,
used where the mechanism resolves the pair jointly rather than by local
readout of λ.

## 3. RULE — Born from no-signaling, internally (C3-B)
**Steering is generated, not imported.** With μ_χ, χ>0 yields steering
diversity D(χ): 0.03→0.08 as χ grows (C3B_hardening.json, 968aaa5) — remote
ensembles with **equal means, distinct higher moments**.

**Selection lemma (executable).** For the readout as a random variable over
the conditioning, no-signaling ⇔ the readout is affine in the projection
(Jensen gap of an affine map ≡ 0). Affinity ∩ oddness (f(−c)=1−f(c)) ∩
normalization = the single rule f₂=(1+c)/2 ⇒ **p=2 uniquely**. The
algebraic zero at p=2 is exact (addendum2, ccf5af6); the non-affine tail is
strictly positive — scan Δ(4)=0.042, Δ(6)=0.105, Δ(10)=0.217, Δ(∞)=0.375,
monotone, analytic↔numeric <2σ (C3L_L3.json, 9567ddb).

**Hardened to theorem.** T1 (sequential/order, B7): repeatability 1.0000,
order commute, no-signaling both orders, post-measurement null-set {2}.
T2 (anti-circularity): **five** measure deformations (μ_κ, κ∈{0.5,2,3};
two cubic), all D>0, analytic null-set {2}, |analytic−numeric|<2σ. T3 (F1
direct law) is a boundary (postulated, not steering-collapse). Verdict:
**internal derivation of the Born rule from no-signaling within the ribbon's
steering class** (C3B_hardening.json, c152c98). One **false stop** en route
— a quadrature-tolerance gate absent from preregistration — was caught,
documented, and corrected without loosening any registered criterion
(addendum2, ccf5af6; §7).

**Mechanical realization (C3-B-mech, form-free).** The steering premise is
realized mechanically and form-free: an annealed, field-magnetized and
fast-frozen elastic source supplies a polar-biased orientation measure that
fits no registered closed form; no closed form is needed. The sign-readout
conditionals of that measure carry steering diversity in the third and fifth
moments, matching the Jensen analytics computed directly on the empirical
ensembles, with the unbiased (h=0) control showing null effects throughout; on
ensembles produced entirely by the elastic dynamics, no-signaling selects p=2.
The selection theorem's premise thus rests on measured moments, not on an
assumed measure — a weaker assumption and hence a stronger claim
(C3Bmech_M1M2.json, 2080bbe; M1✓M2✓ in form-free mode). That the source measure
admits no closed form is kin to the C2-TM wall (the relaxational amplitude
A(k_f) likewise has no closed form): an observation — elasticity does not favor
closed forms — not a theorem.

## 4. CEILING — the 1/3-cosine point
Local resolution (hidden λ + local linear readout) yields
E(a,b)=E_λ[(λ·a)(λ·b)]=aᵀ(⅓I)b = **cosθ/3** (χ-corrections odd ⇒ vanish),
S=2√2/3≈0.943 — the isotropic-cosine ceiling of the local layer. Reached
independently three ways: dynamical isotropization ρ=0.335 (C2ISO_analysis.json,
a043f8f); the analytic lever rule; frame-local space-like resolution
E(0)=0.333, E(90)=0, E(180)=−0.333 (C3L_L2.json, 49e8c1b). Three ideas, one
point ⇒ a structural boundary, the lower bank of the seam.

## 5. SEAM — the trilemma of horns (C3-S)
Can an internal model reach **D>0 and S=2√2** together? Registered attempts,
ansätze fixed before batteries:
- **S-F1** (weighted chord, α=√(1+χ), β=√(1−χ)): S(χ)=2√2·√(1−χ²) —
  reaches 2√2 only at χ=0 (symmetry), but is postulated (no continuous
  measure) ⇒ D_joint N/A. **Horn: no internal steering.**
- **S-F2** (shifted source × chord/product mixture): S(χ)=√2·(1+χ/3)≤1.886,
  and P(s|a,λ)=½ ⇒ Alice does not update λ ⇒ D_joint≈0. **Horn: no
  amplitude / no steering** (C3S.json, 324f097).
- **S-F3** (local bias × chord kernel, |·| ansatz with κ): steers (D>0) but
  the marginal carries Bob-setting dependence at O(κχ) — telegraph
  0.500→0.542 (C3S_F3_check.py; analytics 2053106). **Horn: superluminal
  telegraph** — kin to finite-speed no-gos [BPA12].

**Reduction (Bell corollary, not a new theorem).** The internal class =
λ-average of local responses = LHV ⇒ S≤2. Closing the seam therefore demands
**joint, nonlocal resolution**; and joint resolution demands a preferred
frame (§6). Attempt count: 3. (C3S_seam_reduction.md.)

## 6. FRAME — foliation (C3-L, L1/L2)
On the chord joint law, two resolution mechanics:
- **(i) preferred-foliation:** S=2√2 **identical across five foliations**
  (cross-disc 0.000000 under common random numbers), foliation statistically
  invisible (C3L_L2.json, 49e8c1b).
- **(ii) frame-local:** space-like → LHV ceiling S=0.943; time-like A→B, B→A →
  reconstruct 2√2.
L1: the joint admits an order-invariant realization (existential PASS,
factorized witness disc=0.00); the collapse realization's order-dependence
(0.0418 with invariant marginals) is evidence for L2c, not pathology
(addendum1). **Method note:** the foliation-invariance test required common
random numbers — a raw cross-range would false-STOP on Monte-Carlo noise (§7).

**Conjecture L2c** (status: conjecture with model evidence, path α/β/γ):
outcome-definiteness (P1) + joint resolution (P2) + S=2√2 ⇒ a preferred,
statistically hidden foliation. **Experimental anchor:** real before-before
experiments [ZBGT01, SZGS02] exclude the frame-local branch (ii) in nature;
the surviving corner is preferred-foliation (i) — exact QM, invisible frame
— consistent with L2c and with Bohmian preferred slicing [DGZ92]; kin to
[H92], [SS97].

## 7. METHODS — batteries, controls, self-correction
All batteries: pure numpy, N=2·10⁶, PRNGKey from config, analytics beside
numerics (>2σ mismatch = stop), assert prereg↔JSON before raw, raw→commit→
analysis. Estimator named in every prediction; metric = the model observable.

**Registered false stops and their resolution (part of the evidence):**
1. **Δ(2)>0 = bug** (C3-B): normalized conditional axis broke total
   probability; fixed to raw Bloch (968aaa5).
2. **Quadrature gate** (hardening): an analytic-tolerance gate (tol=1e-9 on
   a 20001-pt trapezoid, error ~2.3e-5) false-STOPped T1/T2; corrected to the
   registered <2σ criterion + algebraic zero at p=2; threshold **not** loosened
   to pass (addendum2, ccf5af6).
3. **Monte-Carlo cross-range** (L2): raw 5-foliation range would false-STOP;
   replaced by common-random-number comparison (49e8c1b).
4. **§2.3 over-claim retraction** (C3-S): the first §2.3 asserted "no-signaling
   alone cuts p>2 in the joint layer"; J5 rediscovered the Popescu–Rohrlich
   fact that the postulated |·|^p family is no-signaling up to the PR box
   [PR94]; the claim was narrowed to steering-endowed layers (324f097; §8).
Each was caught by a preregistered criterion, not by hindsight.

## 8. The Tsirelson section (§2.3, corrected — verbatim)
> Rule selection is exactly as strong as steering. In any layer whose
> measurements produce conditional states that can themselves be steered —
> internally generated diversity D>0 — the Jensen argument applies:
> no-signaling forces affinity, selects p=2, and with it the quantum value
> of S available to that layer. In a bare-correlation joint layer, however,
> where outcomes carry no post-measurement structure, the |·|^p family has
> uniform marginals for every p: the entire family is no-signaling, up to and
> including the PR box — the classic observation of Popescu and Rohrlich
> [PR94], which our battery rediscovered as a registered finding against an
> earlier over-claim of this section (retained in the record). No-signaling
> alone therefore does not enforce the Tsirelson bound; what enforces it, in
> this geometry, is no-signaling PLUS steering structure — and this localizes,
> within one construction, the same gap that information causality [Paw09] was
> invented to fill. The amplitude seam then resolves into a reduction rather
> than a mystery: within the internally-generated class (hidden measure plus
> local response) Bell's theorem itself caps S at 2, so closing the seam —
> D>0 together with S=2√2 — demands joint, nonlocal resolution of the pair;
> and joint resolution, by the foliation results above, demands a preferred
> frame that the statistics provably hide. The program's bridge statement
> follows: Einstein's prohibition selects the quantum rule wherever mechanisms
> carry steering; the quantum amplitude is bought only by joint resolution,
> whose frame Einstein's own statistics render invisible.

## 9. Outlook (verbatim)
> Three cycles of this program asked one question from three sides: what
> stands between a mechanism and quantum correlations. The answer assembled
> itself into four registered statements. (1) RULE: wherever a mechanism
> carries steering — conditional states with internally generated diversity —
> the prohibition of superluminal signaling forces the Born exponent p=2
> uniquely; the derivation is internal, executable, and robust to deformations
> of the source measure. (2) CEILING: mechanisms that resolve outcomes locally
> (hidden measure plus local response) are capped by Bell's theorem at S=2,
> with the 1/3-cosine point as their isotropic-cosine ceiling — a landmark our
> program reached independently from simulation (relaxational dynamics), from
> analysis (the lever rule), and from frame-local resolution. (3) SEAM: the
> gap between that ceiling and the quantum value is not crossed by decorating
> local mechanisms — our registered attempts fail on a trilemma of horns (no
> steering, no amplitude, or superluminal telegraph; kin to finite-speed
> no-gos [BPA12]) — it is crossed only by joint resolution of the pair.
> (4) FRAME: joint resolution of the frame-local kind reconstructs the
> Suarez–Scarani alternative and degrades to the local ceiling, which real
> before-before experiments [ZBGT01, SZGS02] have excluded in nature; the
> surviving corner, preferred-foliation resolution, reproduces quantum
> statistics exactly while rendering its own foliation statistically invisible
> to the resolution of our batteries. The bridge statement of the program is
> therefore not a unification but an exact division of labor: Einstein's
> causality selects the quantum rule; the quantum amplitude is bought only by
> joint resolution; and the frame that joint resolution requires is precisely
> what the selected statistics forever hide. What remains open is stated as
> precisely: a mechanical realization of the asymmetry parameter (C3-B-mech),
> the structural theorem behind the seam trilemma (conjecture L2c, path
> α/β/γ), and the ground-state problem of the stiff chain — walls mapped,
> kill criteria attached, commits public.

## Appendix A — batteries (reproducible)
| Battery | Prereg | Raw | Result |
|---|---|---|---|
| C3-B B1–B7 | dd7a74e | 968aaa5 | steering generated, p=2 forced |
| C3-B hardening T1–T3 | 96a3178 | c152c98 | internal derivation (T1✓T2✓) |
| L3 scan + bridge | 2d78c51 | 9567ddb | tail Δ>0; internal S(2)=√2 |
| C3-L L1/L2 | 0b10f5d,100a261 | 1e0cf0a,49e8c1b | foliation invisible; frame-local→ceiling |
| C3-S J1–J7 | 3d516d4 | 324f097 | seam not closed (S-F1/S-F2) |
| S-F3 analytics | — | 2053106 | closed pre-battery (telegraph) |

## References
[H92] L. Hardy, Phys. Rev. Lett. 68, 2981 (1992).
[SS97] A. Suarez, V. Scarani, Phys. Lett. A 232, 9–14 (1997).
[ZBGT01] H. Zbinden, J. Brendel, N. Gisin, W. Tittel, Phys. Rev. A 63, 022111 (2001).
[SZGS02] A. Stefanov, H. Zbinden, N. Gisin, A. Suarez, Phys. Rev. Lett. 88, 120404 (2002).
[DGZ92] D. Dürr, S. Goldstein, N. Zanghì, J. Stat. Phys. 67, 843 (1992).
[Paw09] M. Pawłowski et al., Nature 461, 1101 (2009).
[BPA12] J.-D. Bancal, S. Pironio, A. Acín, Y.-C. Liang, V. Scarani, N. Gisin, Nature Phys. 8, 867 (2012).
[PR94] S. Popescu, D. Rohrlich, Found. Phys. 24, 379 (1994).
[G89] N. Gisin, Helv. Phys. Acta 62, 363 (1989).
[G90] N. Gisin, Phys. Lett. A 143, 1 (1990), DOI 10.1016/0375-9601(90)90786-N.
[G91] N. Gisin, Phys. Lett. A 154, 201 (1991).
[P91] J. Polchinski, Phys. Rev. Lett. 66, 397 (1991).
[S35] E. Schrödinger, Proc. Camb. Phil. Soc. 31, 555 (1935); 32, 446 (1936).
[HJW93] L. P. Hughston, R. Jozsa, W. K. Wootters, Phys. Lett. A 183, 14 (1993).
[SBG01] C. Simon, V. Bužek, N. Gisin, Phys. Rev. Lett. 87, 170405 (2001).
[A04] S. Aaronson, quant-ph/0401062 (2004).

## Missing ARCH-checks (для сверки архитектором, DOI при вёрстке)
- CDP-родня по стиринг-аксиоматике (Chiribella–D'Ariano–Perinotti) — не в списке, кандидат для §6.
- Bell 1964 (Physics 1, 195) — добавить в список формально (цитируется в §5 редукции).
- финальная DOI-сверка всех канонических [S35,HJW93,SBG01,A04] при вёрстке.
