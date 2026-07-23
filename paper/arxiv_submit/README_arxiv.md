# arXiv submission package — build & font notes

## Upload set (exactly these 8 files)
- `c3_submission.tex` — standalone LaTeX (pandoc→XeLaTeX output of
  `../C3_paper_DRAFT_v1.md`); self-contained (no `\input`/`\include`/graphics/bib).
- 7 DejaVu TrueType fonts, shipped alongside the source so the build does NOT
  depend on DejaVu being installed on arXiv's host:
  `DejaVuSerif.ttf`, `DejaVuSerif-Bold.ttf`, `DejaVuSerif-Italic.ttf`,
  `DejaVuSerif-BoldItalic.ttf`, `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`,
  `DejaVuSansMono.ttf` (~3.2 MB total).

`metadata.md` and `font_fallbacks.md` are documentation — do NOT upload.
Put the fields from `metadata.md` into arXiv's web form (title / abstract /
category quant-ph / comments).

## Build (XeTeX engine; control build passed)
The source uses `\setmainfont{DejaVuSerif}[Path=./, Extension=.ttf, …]` etc.,
i.e. **Path-based** references to the shipped TTFs — no fontconfig name lookup,
so the result is identical regardless of the host's font install.
```bash
xelatex c3_submission.tex   # or: tectonic -X compile c3_submission.tex
```
Clean-dir control build (tectonic, only the .tex + 7 TTFs present):
**0 errors, 0 "Missing character", 0 non-reproducible absolute-path warnings,
PDF 83808 B**, byte-identical to the name-lookup build.

## Why DejaVu (not Latin Modern)
The paper uses Unicode math/Greek in TEXT mode (θ σ √ ⊗ ⅓ ᵀ ½ χ λ ρ α ε …).
DejaVu Serif covers them all (0 gaps). The Latin-Modern-only variant was tested
and has **~184 glyph gaps**, so a broad Unicode text font is required; shipping
the TTFs is the safe, host-independent way to guarantee it.

## Reproducibility line (for the arXiv comments field)
"Fully reproducible: every claim carries a commit hash; batteries are single
numpy scripts. DOI 10.5281/zenodo.21505219"
