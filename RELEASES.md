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

## v2.1 — The tripartite mirror, and a submission (release)
Git tag `v2.1`; GitHub Release with the paper PDFs. Version DOI:
`10.5281/zenodo.21505219` (concept DOI `10.5281/zenodo.21383667`, all versions).
Freezes the submission state: paper 3 in submission form (method frame forward,
three theorem-status results: Born selection / class-M precedence /
schedule-invisibility), cycle-4 synthesis, publication kit (arXiv source +
journal cover letter + endorser targets).

## v2.2 — Submission snapshot (release)
Git tag `v2.2` (commit `e7b2273`); GitHub Release with five PDF assets, among
them `c3_SCIPOST_submission_v3.pdf` — paper 3 exactly as submitted. Version DOI:
`10.5281/zenodo.21651899` (concept DOI `10.5281/zenodo.21383667`, all versions).
Freezes the state the submitted manuscript points at: paper 3 with its new
"Data and code availability" section and a cleaned title block, the arXiv kit in
its host-independent font variant, and the publication correspondence out of the
public tree. No new physics since v2.1.

Two rules were executed here for the first time. **DOI loop broken:** the
manuscript names the release and the *concept* DOI (which resolves to the latest
version) instead of a version DOI minted before the snapshot existed — so no
future snapshot can cite a DOI older than itself, and the paper needs no edit
when a version DOI appears. **Self-contained source archive:** the tag carries
`RELEASE_NOTES_v2.2.md` inside it, so the Zenodo source zip documents its own
contents without reference to anything outside the archive.

## v2.3 — Disclosure, figures, and a working sync gate (release)
Git tag `v2.3`; GitHub Release with the paper PDFs. Version DOI pending until
Zenodo archives the tag (concept DOI `10.5281/zenodo.21383667`, all versions).
**No new physics since v2.2** — no measured number, verdict, or claim changes.
The three new figures plot values already published in v2.1/v2.2 and cite the
same raw files and commits.

AI disclosure harmonized across both papers: one vocabulary in article text
(*the author* for scientific decisions, kill criteria, veto, reproduction; *AI
architect role* for proposals and drafts; *AI executor role* for implementation
and runs), a three-way contradiction in paper 3 over who re-executed C3-B and
the β-coin resolved to the author, a disclosure section added to paper 3 where
there was none, and paper 1 §9.2 rewritten because "used for research planning
and experimental design" read as though the AI designed the study. Disclosure
volume rose (`a7cf54b`). Three figures added, each caption naming its protocol
and provenance (`8848956`). **DOI loop broken in paper 1** (`813d7c5`): §9.1
now names the release and the concept DOI, the rule fixed at v2.2 and until now
applied only to paper 3.

Two gates were built here and both have been shown to fail on demand, not only
to pass. **Captions inside the sync perimeter:** the `figure` environment is no
longer stripped, so the protocol line of each trilemma edge — the sentence that
stops the DS3 and ISO-DYN edges from reading as a contradiction — is compared
across all five sources. **L0, the release label across both papers:** merging
the disclosure and figure branches produced a conflict git resolved cleanly and
wrongly, leaving one source on v2.2 while four said v2.3; no textual merge can
see that, so the label is now checked across paper 3's five sources and paper 1's
availability source and generated FINAL. `paper/check_sync.py` exits 0 for the
first time since it was written — the three long-standing mismatches were both
copies of `c3_submission.tex` missing the availability section, and a source of
record that was gitignored and therefore drifted across branches (`100baf9`).
