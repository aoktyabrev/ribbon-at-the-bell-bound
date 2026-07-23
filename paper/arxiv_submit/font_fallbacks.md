# arXiv font fallbacks — VARIANT (a) IS APPLIED

> **STATUS (applied):** Variant (a) below is now the committed state —
> `c3_submission.tex` references the 7 shipped DejaVu `.ttf` files by Path, and
> those TTFs live in this directory. Clean-dir control build: 0 errors, 0 missing
> characters, 0 absolute-path warnings, PDF 83808 B. Variant (b) is kept only as
> a documented alternative. The historical "prepared, not applied" note below is
> retained for the record.

---

# arXiv font fallbacks — PREPARED, NOT APPLIED (operator decides)

## Situation
`c3_submission.tex` uses fontconfig NAME lookup: `\setmainfont{DejaVu Serif}`
(+ `\setsansfont{DejaVu Sans}`, `\setmonofont{DejaVu Sans Mono}`), with
`unicode-math` loaded by the pandoc template. The paper uses Unicode glyphs in
TEXT mode (θ σ √ ⊗ ⅓ ᵀ ½ χ λ ρ α ε …); DejaVu Serif covers them all.

**Local build (tectonic, this machine, DejaVu installed): 0 errors, 0 "Missing
character".** DejaVu Serif also ships in arXiv's TeX Live (`texlive-fonts-extra`)
and is *usually* present — but NOT guaranteed. If arXiv's XeLaTeX cannot find
"DejaVu Serif", the name lookup fails. Two fallbacks below.

---

## Variant (a) — ship the DejaVu TTFs (SAFE, recommended)
Guarantees byte-identical rendering to the tested build (same fonts, embedded).

1. Copy the font files into this directory (next to the .tex):
```
cp /usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf \
   /usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf \
   /usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf \
   /usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf \
   /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
   /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
   /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf \
   paper/arxiv_submit/
```
2. Replace the three `\setmainfont/\setsansfont/\setmonofont` lines with Path-based:
```latex
\setmainfont{DejaVuSerif}[Path=./, Extension=.ttf,
  BoldFont=DejaVuSerif-Bold, ItalicFont=DejaVuSerif-Italic,
  BoldItalicFont=DejaVuSerif-BoldItalic]
\setsansfont{DejaVuSans}[Path=./, Extension=.ttf, BoldFont=DejaVuSans-Bold]
\setmonofont{DejaVuSansMono}[Path=./, Extension=.ttf]
```
Upload = `c3_submission.tex` + the 7 `.ttf` files (~2.4 MB). arXiv XeLaTeX
accepts TTFs shipped in the source. Rendering identical to the tested PDF.

---

## Variant (b) — TeX Gyre (TeX Live standard, guaranteed present, UNTESTED here)
No font files to ship; uses fonts always in TeX Live. But TeX Gyre Termes may
NOT cover every Unicode TEXT glyph the paper uses (⊗ ⅓ ᵀ …) → risk of new
"Missing character". **Could not test locally** (TeX Gyre absent on this box, no
system TeX Live) — MUST be compiled on TeX Live / the arXiv sandbox to confirm
0 missing characters before trusting it.

Replace the three `\set*font` lines with:
```latex
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}
\setmathfont{TeX Gyre Termes Math}   % unicode-math needs a math font
```
If a glyph is missing under (b), either add a covering fallback font for that
range or fall back to (a).

---

## Recommendation
**(a) is the safe choice** — it ships the exact fonts that produced the clean
build, so arXiv renders identically regardless of its font install, at the cost
of ~2.4 MB of TTFs in the upload. Use (b) only if you prefer a fontless upload
AND have verified 0 missing characters on TeX Live first.

Nothing here is applied to `c3_submission.tex`; the committed source is the
name-lookup version (clean where DejaVu is installed).
