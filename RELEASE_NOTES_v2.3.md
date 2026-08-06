# Release v2.3 — "Disclosure, figures, and a working sync gate"

**Version DOI: pending** — minted by Zenodo when this tag is archived, then
closed into README and RELEASES.md, as at v2.0/v2.1/v2.2.
Concept DOI `10.5281/zenodo.21383667` — all versions, resolves to the latest.

**No new physics since v2.2.** Nothing in this release changes a measured
number, a verdict, or a claim. The three new figures look like results but are
not: they plot values already published in v2.1/v2.2 and cite the same raw files
and commits. What this release contains is precision of statement, two figures'
worth of legibility, and machinery that makes a class of silent error loud.

## What's new since v2.2

- **AI disclosure harmonized across both papers.** One vocabulary in article
  text: *the author* for scientific decisions, kill criteria, veto, and
  reproduction in a separate environment; *AI architect role* for proposals,
  drafts, and internal review; *AI executor role (Claude Code)* for
  implementation, runs, and source checks. Bare "operator"/"architect" no longer
  appear in either manuscript (`a7cf54b`).
  - Paper 3 §2 and §8 disagreed with each other and with §DA on who re-executed
    C3-B and the β-coin — *operator*, *architect*, *author* in three places.
    Resolved to the author, per the author's own answer.
  - Paper 3 gains a **Use of generative AI** section before Data availability;
    it had none.
  - Paper 1 §9.2 said the tools were "used for research planning and
    experimental design", which reads as though the AI designed the study.
    Replaced with the explicit three-party division and the author's ownership
    of the research question, kill criteria, and veto.
  - Disclosure **volume increased**, not decreased: §9.2 of paper 1 went 80 → 129
    words, paper 3 gained 154. SI-M.1, §8.6, and the false-stop and retraction
    registers are untouched.

- **Three figures, all plotting already-published numbers** (`8848956`).
  - Paper 3 §5, **Fig. 1 — the CHSH landscape**: the 1/3-cosine ceiling
    2√2/3, the measured relaxation family S = 2ρ, the Bell and Tsirelson bounds,
    and the algebraic maximum. The caption states that the 4.0 is the CHSH
    maximum at N = 2 (the PR box) and **not** the Mermin–Klyshko 4.0 at N = 4 —
    the operator confusion the cycle-4 climbing-ladder retraction was made of
    (`ff28341`).
  - Paper 3 §5, **Fig. 2 — the trilemma**, with the protocol of each edge named.
    Two edges isotropize and they are not the same operation: DS3 applies the
    Haar rotation post hoc to a frozen source with settings independent of λ, so
    Bell's theorem forces the triangle; ISO-DYN rotates the preparation before
    relaxation with the settings entering as clamps, so a cosine is available and
    is paid for in amplitude. Both pre-registered caveats are on the figure: the
    isotropy is Haar-*averaging* of an anisotropic response, not a symmetry of
    the model, and ISO-DYN *brackets* the 1/3-cosine point rather than landing
    on it.
  - Paper 1 §6.5, **Fig. 5 — the three axes of failure**. Phase D populates two
    of the three property pairs; the third corner is drawn **empty and labelled
    empty**, because the pair (isotropy + cosine form) is realized only by
    cycle-2 ISO-DYN, outside this paper's evidence base. The emptiness is part
    of the result.
  - Every plotted number is read from frozen analysis JSON; only literature and
    analytic constants are typed in. Scripts committed: `plot_chsh_landscape.py`,
    `plot_trilemma_isodyn.py`, `plot_trilemma_axes.py`.

- **The sync gate works, and passes for the first time** (`100baf9`, `158fe93`,
  `813d7c5`). `paper/check_sync.py` guards paper 3's five hand-maintained
  sources, which have no generator.
  - Both copies of `c3_submission.tex` were missing the **Data and code
    availability** section entirely — three anchors had been out of sync since
    the check was written. Inserted; the arXiv kit needed it regardless.
  - `private_upload/arxiv_paper3/c3_submission.tex` is a source of record for the
    check but was gitignored, so branch switching never reverted it: the copy
    accumulated one branch's edits while being compared against another's. Now
    tracked. `private_correspondence/` stays out of the public tree entirely.
  - **Figure captions brought inside the comparison perimeter**: the `figure`
    environment is no longer stripped, and `\includegraphics` and markdown images
    are dropped symmetrically on both sides. The protocol line of each trilemma
    edge is exactly the text that must not drift.
  - **L0, the release label, across both papers.** Merging the disclosure and
    figure branches produced a conflict git resolved cleanly and wrongly: the
    availability section arrived in one source still saying v2.2 while four
    others said v2.3. No textual merge can see that. The label is now checked
    across paper 3's five sources *and* paper 1's availability source and
    generated FINAL — two pipelines sharing one fact, connected until now only
    by a human remembering.

- **DOI loop broken in paper 1 too** (`813d7c5`). §9.1 cited version DOI
  `10.5281/zenodo.21388902` (paper-v1.2) — a snapshot this release does not
  describe. It now names the release and the concept DOI only, the rule fixed at
  v2.2 and until now applied only to paper 3. Paper 1 needs no availability edit
  at any future release.

- **Paper 1's PDF layer builds without tectonic** (`f028f3f`). `build_pdf.sh`
  falls back to xelatex when tectonic is absent; both are XeTeX, the difference
  is only that tectonic fetches missing packages itself (`tlmgr install soul`
  covers the gap, no sudo). Two bugs in that fallback were found by reading the
  built PDF rather than the exit code: `return` inside a subshell is not a
  function return, and `-output-directory` puts the output directory on the
  *input* search path — xelatex silently read a stale `paper/pdf/main.tex` from
  15 July and built `main.pdf` from draft content, 23 pp, exit code 0, success
  message printed. `compile_tex` now clears that path before every run.

- **A claim about this repository that had become false.** §9.1 of paper 1
  states that every figure can be regenerated from a named commit by running its
  committed script, and then listed the scripts for Fig. 1–4 only. Fig. 5 was
  missing from the list. Caught by reading the rebuilt PDF (`813d7c5`).

## Assets

- `main.pdf` — paper 1, camera-ready, **22 pp** (was 21): Fig. 5 added in §6.5,
  §9.2 rewritten, §9.1 on the concept DOI.
- `si.pdf` — paper 1 supplement, 3 pp; SI-M.1 unchanged.
- `c3_fop_source_pdflatex.pdf` — paper 3, 9 pp (was 7): the AI disclosure
  section and the two §5 figures.
- `c2_synthesis_TR.pdf` — cycle-2 synthesis, technical report.

## Reproducibility

Every quantitative claim carries a commit hash; batteries are single numpy (or
JAX/GPU) scripts run under pre-registered kill criteria; the cycle reproduces
bitwise. Figures regenerate from committed scripts against committed raw data.
`python paper/check_sync.py` exits 0. Repository:
<https://github.com/aoktyabrev/ribbon-at-the-bell-bound>.
