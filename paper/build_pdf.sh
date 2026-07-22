#!/usr/bin/env bash
# Сборка PDF: markdown → (pandoc) → LaTeX → (tectonic/xelatex) → PDF.
#
# DRAFT_v<V>.md → main.pdf ; SI_v<V>.md → si.pdf
#
# Движок — XeTeX: корпус содержит юникод-математику (ℤ₂ θ σ × → ⟨⟩ π₁ R⁴ √ ≈ ≤ − ∈ Σ ∝ ∞)
# И кириллицу (цитаты первоисточников, названия разделов канона). Шрифт DejaVu покрывает
# оба набора — макросов-замен не требуется, xelatex берёт юникод как есть.
#
# Инструменты ставятся БЕЗ sudo:
#   pandoc    — uv pip install --python sim/.venv/bin/python pypandoc-binary
#   tectonic  — статический бинарь с github releases (см. TECTONIC ниже)
#
# Запуск: bash paper/build_pdf.sh [версия]   (по умолчанию v3)

set -euo pipefail

V="${1:-v3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$HERE/pdf"

PANDOC="${PANDOC:-$ROOT/sim/.venv/lib/python3.12/site-packages/pypandoc/files/pandoc}"
TECTONIC="${TECTONIC:-tectonic}"

command -v "$TECTONIC" >/dev/null 2>&1 || [ -x "$TECTONIC" ] || {
  echo "tectonic не найден. Задай TECTONIC=/путь/к/tectonic" >&2; exit 1; }
[ -x "$PANDOC" ] || { echo "pandoc не найден: $PANDOC" >&2; exit 1; }

mkdir -p "$OUT"

# Глифы, которых нет в DejaVu Serif (проверено fontTools по cmap шрифта):
#   U+2113 ℓ → \ell   U+226A ≪ → \ll
# Остальная юникод-математика (ℤ θ σ × → ⟨⟩ π ₁ ₂ ⁴ √ ≈ ≤ − ∈ Σ ∝ ∞ Δ χ ² ° ê λ ρ α)
# и кириллица шрифтом покрыты — xelatex берёт их как есть.
preprocess() {
  # Глифы, которых нет в DejaVu Serif — заменяем на матрежим-эквиваленты (amssymb).
  # ≪/≫ → текстовые <</>> (инъекция $...$ в плотных таблицах с ^/_ ломает
  # парность мат-режима и утягивает кириллицу в lmmi10). ℓ — только в мат-контексте.
  sed -e 's/ℓ/$\\ell$/g' -e 's/≪/<</g' -e 's/≫/>>/g' \
      -e 's/✓/$\\checkmark$/g' -e 's/✗/$\\times$/g' "$1"
}

build() {
  local src="$1" name="$2" title="$3"
  echo "=== $name: $src"
  # markdown → LaTeX. resource-path — чтобы ../sim/phase_D/fig/*.png нашлись.
  preprocess "$HERE/$src" | "$PANDOC" \
    --from=markdown-yaml_metadata_block-simple_tables-multiline_tables-grid_tables-implicit_figures+footnotes+pipe_tables+strikeout+tex_math_dollars \
    --to=latex \
    --standalone \
    --resource-path="$HERE:$ROOT" \
    --variable=documentclass:article \
    --variable=classoption:11pt \
    --variable=geometry:a4paper,margin=25mm \
    --variable=mainfont:"DejaVu Serif" \
    --variable=sansfont:"DejaVu Sans" \
    --variable=monofont:"DejaVu Sans Mono" \
    --variable=colorlinks:true \
    --variable=linkcolor:black \
    --variable=urlcolor:blue \
    --variable=title:"$title" \
    --variable=header-includes:"\\usepackage{amssymb}\\usepackage[normalem]{ulem}\\let\\st\\sout ${EXTRA_HEADER:-}" \
    --pdf-engine=xelatex \
    --output="$HERE/$name.tex"

  # LaTeX → PDF. tectonic тянет пакеты сам, сети хватает одного прогона.
  ( cd "$HERE" && "$TECTONIC" -X compile "$name.tex" --outdir "$OUT" --keep-logs ) \
    || { echo "СБОРКА $name УПАЛА — см. $OUT/$name.log" >&2; return 1; }

  rm -f "$HERE/$name.tex"
  echo "  → $OUT/$name.pdf  ($(du -h "$OUT/$name.pdf" | cut -f1), \
$("$PANDOC" --version >/dev/null; pdfinfo "$OUT/$name.pdf" 2>/dev/null | awk '/^Pages/{print $2" стр."}' || echo "стр.: ?"))"
}

# Диспетч целей: v3 (по умолчанию) | FINAL | C3DRAFT | C2TR
case "$V" in
  FINAL)
    # paper 1 финальная редакция → main.pdf + si.pdf (SI — единственный SI_v3)
    build "FINAL_v1.md" main "The Ribbon at the Bell Bound"
    build "SI_v3.md"    si   "Supplementary — The Ribbon at the Bell Bound"
    echo; echo "Готово (FINAL): $OUT/main.pdf, $OUT/si.pdf" ;;
  C3DRAFT)
    # paper 3 драфт → c3_draft_v1.pdf, водяная честность в колонтитуле
    HASH="$(cd "$ROOT" && git log -1 --format=%h -- paper/C3_paper_DRAFT_v1.md)"
    DATE="$(cd "$ROOT" && git log -1 --format=%cs -- paper/C3_paper_DRAFT_v1.md)"
    EXTRA_HEADER="\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\cfoot{DRAFT v1 — ${DATE} — commit ${HASH}}\\rfoot{\\thepage}\\lfoot{cycle-3 paper}" \
      build "C3_paper_DRAFT_v1.md" c3_draft_v1 "Born from causality, Tsirelson from steering, and the amplitude seam (DRAFT v1)"
    echo; echo "Готово (C3 draft): $OUT/c3_draft_v1.pdf" ;;
  C3DRAFTV2)
    # paper 3 драфт v2 → c3_draft_v2.pdf, колонтитул v2 + актуальный хэш
    HASH="$(cd "$ROOT" && git log -1 --format=%h -- paper/C3_paper_DRAFT_v1.md)"
    DATE="$(cd "$ROOT" && git log -1 --format=%cs -- paper/C3_paper_DRAFT_v1.md)"
    EXTRA_HEADER="\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\cfoot{DRAFT v2 — ${DATE} — commit ${HASH}}\\rfoot{\\thepage}\\lfoot{cycle-3 paper}" \
      build "C3_paper_DRAFT_v1.md" c3_draft_v2 "Born from causality, Tsirelson from steering, and the amplitude seam (DRAFT v2)"
    echo; echo "Готово (C3 draft v2): $OUT/c3_draft_v2.pdf" ;;
  C3SUBMIT)
    # paper 3 submission-финал → c3_draft_v3.pdf, колонтитул SUBMISSION v3 + хэш
    HASH="$(cd "$ROOT" && git log -1 --format=%h -- paper/C3_paper_DRAFT_v1.md)"
    DATE="$(cd "$ROOT" && git log -1 --format=%cs -- paper/C3_paper_DRAFT_v1.md)"
    EXTRA_HEADER="\\usepackage{fancyhdr}\\pagestyle{fancy}\\fancyhf{}\\cfoot{SUBMISSION v3 — ${DATE} — commit ${HASH}}\\rfoot{\\thepage}\\lfoot{cycle-3 paper}" \
      build "C3_paper_DRAFT_v1.md" c3_draft_v3 "Born from causality, Tsirelson from steering, and the amplitude seam (SUBMISSION v3)"
    echo; echo "Готово (C3 submission v3): $OUT/c3_draft_v3.pdf" ;;
  C2TR)
    # синтез цикла 2 → технический отчёт
    build "../sim/cycle2/C2_synthesis.md" c2_synthesis_TR "Technical Report — Cycle 2 Synthesis"
    echo; echo "Готово (C2 TR): $OUT/c2_synthesis_TR.pdf" ;;
  *)
    build "${MAIN_SRC:-DRAFT_$V.md}" main "The Ribbon at the Bell Bound"
    build "SI_$V.md"    si   "Supplementary — The Ribbon at the Bell Bound"
    echo; echo "Готово: $OUT/main.pdf, $OUT/si.pdf" ;;
esac
