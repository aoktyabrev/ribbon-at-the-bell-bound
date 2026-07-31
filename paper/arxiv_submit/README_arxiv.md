# arXiv submission package — build & font notes

> **Two kits live in this directory.** `c3_submission.tex` is the **arXiv** kit
> (paper 3, v2.1 era). `c3_fop_source.tex` is the **Foundations of Physics /
> snapp** kit — see "FoP (snapp) kit" at the bottom. They share the 7 TTFs and
> differ in content: only the FoP source carries the "Data and code
> availability" section added at `ebade60`.

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

---

## FoP (snapp) kit — `c3_fop_source.tex`

Foundations of Physics submits through snapp
(`https://submission.nature.com/new-submission/10701/3`), which takes the
manuscript as a ZIP of editable sources and **compiles the PDF on its own
engine**. Springer's guidelines are explicit: "Failing to submit a complete set
of editable source files will result in your article not being considered for
review." A PDF alone is not a submission there.

Upload set: this `.tex` plus the same 7 DejaVu TTFs, **flat in the archive root**
(portal compilers frequently do not resolve nested paths). No PDF, no `.md`, no
documentation files inside the archive.

### Provenance: why the footer says `ebade60`

The footer of this source reads
`SUBMISSION v3 — 2026-07-28 — commit ebade60` and **must not be changed** by
later commits to this file. The program's rule is *hash = commit of the content
source*. `ebade60` is the commit of `paper/C3_paper_DRAFT_v1.md` from which this
LaTeX was generated (pandoc step of `build_pdf.sh C3SUBMIT`, which deletes its
intermediate `.tex`; it was regenerated with identical flags rather than
rebuilt).

The only edit applied on top of that generated source is the font preamble:
fontconfig name lookup (`\setmainfont[]{DejaVu Serif}`) replaced by explicit
Path-based file references (`\setmainfont{DejaVuSerif.ttf}[Path=./, …]`). That
changes **font resolution only** — not one character of the manuscript. Verified
by control build: compiled from the shipped archive in a directory containing
nothing but the `.tex` and the 7 TTFs, the output is **8 pages, 0 errors, 0
"Missing character"**, its text layer is byte-identical to the frozen reference
PDF, and all 8 pages are pixel-identical to it at 150 dpi.

So the footer keeps naming the commit that fixes the *content*, which is what a
reader checking reproducibility needs, and this file's own commit history
records the typesetting-only delta.
