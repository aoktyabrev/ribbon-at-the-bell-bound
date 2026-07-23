# arXiv submission package — build & font notes

## Contents
- `c3_submission.tex` — standalone LaTeX (pandoc→XeLaTeX output of
  `../C3_paper_DRAFT_v1.md`), compiles standalone.
- `metadata.md` — title / abstract (1252 chars, under limit) / category / comments.
- `endorser_targets.md` — endorsement-target authors from the References.

## Build (control build passed)
XeTeX engine. Control build with `tectonic` (bundles fonts) produced a clean PDF
(0 missing characters), byte-comparable to `../pdf/c3_draft_v3.pdf` (minus footer).
```bash
xelatex c3_submission.tex   # or: tectonic -X compile c3_submission.tex
```

## ⚠ FONT REQUIREMENT (verify before submitting)
The source sets `mainfont: DejaVu Serif` — the paper uses Unicode math/Greek in
TEXT mode (θ σ √ ⊗ ⅓ ᵀ ½ χ λ ρ α ε …), which DejaVu Serif covers (0 gaps).
- The arXiv-portable variant (Latin Modern + `unicode-math`, no DejaVu) was
  tested and has **~184 glyph gaps** — Latin Modern does not cover these text
  glyphs. So DejaVu (or an equivalent broad Unicode text font) is REQUIRED.
- arXiv runs TeXLive; DejaVu Serif ships in `texlive-fonts-extra`, usually
  present — but VERIFY on the arXiv sandbox. If absent, upload the DejaVu `.ttf`
  files with the submission (arXiv allows font uploads) or switch every Unicode
  glyph to a LaTeX macro (large edit, not done here).
- `tectonic` here confirms the .tex is correct; the only open item is DejaVu
  availability on arXiv's engine.

## Reproducibility line (for the arXiv comments field)
"Fully reproducible: every claim carries a commit hash; batteries are single
numpy scripts. DOI 10.5281/zenodo.21505219"
