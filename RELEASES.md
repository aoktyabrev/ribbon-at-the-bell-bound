# RELEASES — milestone journal

Chronological. Each milestone: what it established (verbatim from the source
files where a claim is quoted) + the commit(s) that prove it.

## v1.2 — Paper 1 (camera-ready)
"The Ribbon at the Bell Bound." Camera-ready; Zenodo concept DOI
`10.5281/zenodo.21383667` (all versions). arXiv submission in endorsement.
Source: `paper/FINAL_v1.md`. Establishes the cycle-1 amplitude–form–isotropy
trilemma of the geometric hidden-connection model.

## Cycle-2 close — the walls are measured
Cycle 2 established *why* the walls stand: factorization of relaxational readout
(C2-F, F_s<6.3e-4), the rule-dial, quench-glass (C2-J), structural
decorrelation, and the no-closed-form amplitude A(k_f) (C2-TM). Synthesis:
`sim/cycle2/C2_synthesis.md`. Commit `452bff4` ("top-A form закрыт, S≤2").

## C3-B — Born-rule selection theorem
Status **theorem**: "internal derivation of the Born rule from no-signaling
within the ribbon's steering class." The import of partially entangled
ensembles is closed (steering generated internally by a shifted source measure;
p=2 forced by no-signaling; robust under five measure deformations; T3/F1 a
boundary). Battery `968aaa5`; hardening T1/T2/T3 `c152c98`; verdict `552fc82`.

## L-track — the bridge and the class-M theorem
The Tsirelson bound emerges from no-signaling *plus steering structure* (not
no-signaling alone — the §2.3 over-claim was retracted after the battery
rediscovered the PR-box fact, `2053106`). Status **theorem** (class M): S>2 is
achievable iff M admits a shared, space-like-transcending, statistically
invisible precedence structure (canonical statement — in Russian — in
`sim/cycle3/C3L_L2c_THEOREM.md`; rendered here in English, not a verbatim
quote). The frameless per-run coin reaches S=2√2 without signaling and is
invisible. β-coin + third clean adversarial pass: commit `c8e1bf3`. Experimental anchor: real before-before
experiments exclude the frame-local branch in nature (`49e8c1b`).

## C3-B-mech — mechanical steering premise (form-free)
Status **measured**: "the steering premise is realized mechanically and
form-free … the selection theorem's premise thus rests on measured moments, not
on an assumed measure — a weaker assumption and hence a stronger claim." An
annealed, field-magnetized, fast-frozen elastic source supplies a polar-biased
measure with D>0 in the 3rd and 5th moments (h=0 control null), on which
no-signaling selects p=2; the measure fits no closed form. Full run + claim-line
`2080bbe`, `1c653b6`.

## Cycle-4 open — C4-GHZ flagship
Multiparty and connection geometry. Two tracks: G-T (generalize the class-M
theorem to N events; Mermin bounds from global disorder) prior 0.7; G-M
(three-end geometric law reproducing GHZ with invisible precedence, monogamy a
mandatory J-test) prior 0.35; named wall ("bipartite-complete,
tripartite-locked") prior 0.45. Scaffold `990cb99`; canonical inputs `12cb5d6`;
prereg drafts (GM-F1 closed analytically) `812e32a`.

## Cycle-4 in progress — tripartite mirror
Tripartite mirror complete: the seam (signature at the classical bound, GM-F2,
predicted 0.45) and its crossing (the complete GHZ signature AT THE MERMIN
SETTINGS (all marginals ½, zero pairwise, maximal triple, M₃=4) — from
parity-in-geometry plus ONE ordered pair, GM-F2j branch (i)). Scope corrected
after kinship check: full-settings GHZ statistics is genuinely tripartite
(Svetlichny) and unreachable by a fixed ordered pair — architect overclaim,
retracted (self-correction register). Class-M
precedence theorem holds for N=3 (Mermin value); ДЫРА-N2 (general-N sufficiency)
remains open. The G-T2 "climbing depth ladder" was RETRACTED as a non-standard-operator
artifact: under the standard Mermin-Klyshko operator, one ordered pair reaches |M|=2
(the 2-producible ceiling, = quantum at N=3, below quantum for N≥4) and additional
disjoint pairs do not climb (kinship check vs Svetlichny/SS02/CGPRS02).
Schedule-invisibility is now a **theorem** (class M, v2 clean second pass): the linear
extension of the precedence structure is statistically invisible; the structure itself is
certified by achieved tiers (Mermin/Svetlichny), not invisible. Commits: GM-F2 `99205c3`,
GM-F2j `ed8e950`, G-T theorem(N=3) `840602b`, schedule-invisibility theorem `bb0cec6`,
kinship+retraction `C4GT2_kinship_check.md`.

## v2.0 — Cycles 1-3: theorems and walls (release)
Git tag `v2.0`; GitHub Release with four PDF assets (paper 1 camera-ready
main+SI; paper 3 draft v1; cycle-2 technical report). Version DOI:
`10.5281/zenodo.21476426` (concept DOI `10.5281/zenodo.21383667`, all versions).
Freezes the program state: two theorem-status results (C3-B selection; class-M
precedence), the bridge statement, and the public record of false stops.
