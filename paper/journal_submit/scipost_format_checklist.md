# SciPost Physics — format checklist (VERIFY current requirements at submission)
# NB: scipost.org/submissions/author_guidelines returned Access-Denied on
# 2026-07-23 — items below are from prior knowledge; VERIFY each on the live
# portal before submitting.

## To verify on the live SciPost portal
- [ ] **LaTeX class:** SciPost requires the `SciPost.cls` style (from their
      template/GitHub). Our source is pandoc→XeLaTeX (`arxiv_submit/c3_submission.tex`)
      — will need re-typesetting into SciPost.cls (or confirm they accept a
      standalone PDF for first submission and require .cls on acceptance).
- [ ] **Acceptance-criteria / "expectations" statement:** SciPost asks authors to
      state which expectation the paper meets (groundbreaking discovery /
      breakthrough / new pathway / synergetic link). Draft choice: **"open a new
      research pathway"** (reproducible self-correcting reconstruction program).
- [ ] **Suggested referees:** optional field — candidates overlap the arXiv
      endorser list (Scarani, Gisin, Chiribella, Bancal); pick 3, avoid conflicts.
- [ ] **Abstract:** on the submission form (1252 chars, well within any limit).
- [ ] **Data / reproducibility:** SciPost encourages open code/data — point to
      the public repository + Zenodo DOI 10.5281/zenodo.21505219. Strong fit
      (bitwise reproduction, single-script batteries).
- [ ] **Cover letter / author comments:** `cover_letter_DRAFT.md` (this dir).
- [ ] **License:** SciPost is CC-BY-4.0 by default — confirm acceptable.

## Ready in this package
- `cover_letter_DRAFT.md` — assembled from claim-strings (no placeholder).
- Source `../C3_paper_DRAFT_v1.md`; PDF `../pdf/c3_draft_v3.pdf`;
  arXiv .tex `../arxiv_submit/c3_submission.tex`.
