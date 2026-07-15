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
  sed -e 's/ℓ/$\\ell$/g' -e 's/≪/$\\ll$/g' "$1"
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
    --variable=header-includes:'\usepackage[normalem]{ulem}\let\st\sout' \
    --pdf-engine=xelatex \
    --output="$HERE/$name.tex"

  # LaTeX → PDF. tectonic тянет пакеты сам, сети хватает одного прогона.
  ( cd "$HERE" && "$TECTONIC" -X compile "$name.tex" --outdir "$OUT" --keep-logs ) \
    || { echo "СБОРКА $name УПАЛА — см. $OUT/$name.log" >&2; return 1; }

  rm -f "$HERE/$name.tex"
  echo "  → $OUT/$name.pdf  ($(du -h "$OUT/$name.pdf" | cut -f1), \
$("$PANDOC" --version >/dev/null; pdfinfo "$OUT/$name.pdf" 2>/dev/null | awk '/^Pages/{print $2" стр."}' || echo "стр.: ?"))"
}

build "DRAFT_$V.md" main "The Ribbon at the Bell Bound"
build "SI_$V.md"    si   "Supplementary — The Ribbon at the Bell Bound"

echo
echo "Готово: $OUT/main.pdf, $OUT/si.pdf"
