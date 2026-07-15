"""Сборка полного черновика для вето-чтения автора.

Конкатенация sections/ в фиксированном порядке + шапка + список фигур.
Ничего не переписывает в секциях: единственный источник правды — sections/*.md.

Запуск: python paper/build_draft.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECT = os.path.join(HERE, "sections")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
OUT = os.path.join(HERE, f"DRAFT_{VERSION}.md")

ORDER = [
    "00_frontmatter.md",
    "00a_abstract.md",
    "01_intro.md",
    "02_model.md",
    "03_analytical.md",
    "04_methods.md",
    "05_nogo.md",
    "06_results.md",
    "07_selfcorrection.md",
    "08_discussion.md",
    "90_availability.md",
    "99_references.md",
]

# Supplementary — собирается отдельным файлом, в основной корпус не входит
SI_ORDER = [
    "SI_program_history.md",
    "SI_methodology.md",
]

HEADER = f"""<!-- СГЕНЕРИРОВАНО paper/build_draft.py — НЕ РЕДАКТИРОВАТЬ. Правки — в sections/*.md -->

# Full draft {VERSION}

Full draft for author review. Read from the repository file, not from any chat transcript.
Veto points marked in the text: §3.1 [deferred-veto], §8.4 [author veto point: the
"classical route exhausted" position], §8.6 [author veto point: closing credo].

---

"""

FIGURES = """
---

## Figure and table list

All figures are produced by committed scripts from committed raw data; each can be
regenerated from a named commit (Section 9.1).

| # | subject | file | script | section |
|---|---|---|---|---|
| Fig. 1 | plateau A(N) | `d2ext_scaling.png`, `ds2_cross.png` | `analysis_ext.py`, `analysis_ds2.py` | 6.1 |
| Fig. 2 | A_plateau(k_f), stiffness memory | `s1runs_kf.png` | `plot_s1runs_kf.py` | 6.2 |
| Fig. 3 | anisotropy map A(α) | `ds3_aniso.png` | `analysis_ds3.py` | 6.3 |
| Fig. 4 | triangle vs cosine (isotropized) | `ds3_iso.png` | `analysis_ds3.py` | 6.4 |
| Table 1 | CHSH revision: withdrawn vs valid | inline | — | 7.2 |

Figure files live in `sim/phase_D/fig/`.
"""


def words(text):
    """Слова тела: без служебных маркеров, таблиц и путей — грубая, но стабильная мерка."""
    t = re.sub(r"`[^`]*`", " ", text)
    t = re.sub(r"^\s*\|.*$", " ", t, flags=re.M)      # строки таблиц
    t = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", t)      # картинки/ссылки
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9−–—]+", t))


def main():
    parts = [HEADER]
    counts = []
    for name in ORDER:
        with open(os.path.join(SECT, name), encoding="utf-8") as f:
            body = f.read().rstrip() + "\n"
        # пути картинок в секциях относительны sections/; черновик лежит на уровень выше
        body = body.replace("](../../sim/", "](../sim/")
        parts.append(body)
        parts.append("\n---\n\n")
        counts.append((name, words(body)))
    parts.append(FIGURES)

    text = "".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    total = sum(c for _, c in counts)
    print(f"{'секция':<24} {'слов':>6}")
    for name, c in counts:
        print(f"  {name:<22} {c:>6}")
    print(f"  {'ИТОГО (тело)':<22} {total:>6}")
    print(f"\n→ {OUT}")

    # незакрытое — считаем по собранному корпусу, не по памяти
    flags = {
        "[*] (сверка не снята)": len(re.findall(r"\[\*\]", text)),
        "[Fig. N: ...] (плейсхолдер, не вставлена)": len(re.findall(r"^\[Fig\.", text, flags=re.M)),
        "STUB": len(re.findall(r"\[STUB\]", text)),
        "TODO": len(re.findall(r"TODO", text)),
        "pending": len(re.findall(r"pending", text, flags=re.I)),
        "[UNVERIFIED] (references)": len(re.findall(r"\[UNVERIFIED", text)),
    }
    print("\nМаркеры в собранном корпусе:")
    for k, v in flags.items():
        print(f"  {k:<42} {v}")

    # точки вето: считаем по телу (шапка не в счёт) — §2.1 deferred-veto + §7.4/§7.6
    body = text.split("---\n", 1)[1] if "---\n" in text else text
    veto = (len(re.findall(r"deferred-veto: author", body))
            + len(re.findall(r"\[author veto point", body)))
    si_parts = []
    for name in SI_ORDER:
        with open(os.path.join(HERE, "SI", name), encoding="utf-8") as f:
            si_parts.append(f.read().rstrip() + "\n\n---\n\n")
    si_out = os.path.join(HERE, f"SI_{VERSION}.md")
    with open(si_out, "w", encoding="utf-8") as f:
        f.write("".join(si_parts))
    print(f"→ {si_out}")

    print(f"\n  точек вето в теле (ожидается 3: §3.1, §8.4, §8.6): {veto}")


if __name__ == "__main__":
    main()
