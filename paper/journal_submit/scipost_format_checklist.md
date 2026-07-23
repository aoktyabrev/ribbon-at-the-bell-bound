# SciPost Physics — format checklist (VERIFY current requirements at submission)
# NB: scipost.org/submissions/author_guidelines returned Access-Denied on
# 2026-07-23 — items below are from prior knowledge; VERIFY each on the live
# portal before submitting.

## To verify on the live SciPost portal
- [ ] **LaTeX class:** SciPost requires the `SciPost.cls` style (from their
      template/GitHub). The source here is pandoc→XeLaTeX
      (`arxiv_submit/c3_submission.tex`) — re-typesetting into SciPost.cls will
      be needed (or confirm a standalone PDF is accepted for first submission,
      with .cls required on acceptance).
- [ ] **Acceptance-criteria / "expectations" statement:** SciPost asks authors to
      state which expectation the paper meets (groundbreaking discovery /
      breakthrough / new pathway / synergetic link). Candidate: **"open a new
      research pathway"** (reproducible self-correcting reconstruction program).
- [ ] **Suggested referees:** an optional field exists on the submission form.
- [ ] **Abstract:** on the submission form (1252 chars, well within any limit).
- [ ] **Data / reproducibility:** SciPost encourages open code/data — the public
      repository + Zenodo DOI 10.5281/zenodo.21505219 cover this (bitwise
      reproduction, single-script batteries).
- [ ] **License:** SciPost is CC-BY-4.0 by default — confirm acceptable.

## Technical source in this package
- Source `../C3_paper_DRAFT_v1.md`; PDF `../pdf/c3_draft_v3.pdf`;
  arXiv .tex `../arxiv_submit/c3_submission.tex`, metadata `../arxiv_submit/metadata.md`.
- The cover letter and referee/endorser correspondence are maintained privately
  (not in the public kit).
