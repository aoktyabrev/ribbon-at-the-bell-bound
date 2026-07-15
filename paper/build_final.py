"""Camera-ready сборка: sections/ → FINAL_v<N>.md.

Отличие от build_draft.py: тот собирает РАБОЧИЙ черновик для автора (с шапкой ревью,
owner-строками, вето-маркерами, [verified]-пометками). Здесь — версия для читателя:
всё служебное снято, файловые пути заменены на ссылку на репозиторий.

Единственный источник правды — sections/*.md; FINAL генерируемый, править его нельзя.

Запуск: python paper/build_final.py [версия]   (по умолчанию v1)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SECT = os.path.join(HERE, "sections")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"
OUT = os.path.join(HERE, f"FINAL_{VERSION}.md")

REPO = "https://github.com/aoktyabrev/ribbon-at-the-bell-bound"

ORDER = [
    "00_frontmatter.md", "00a_abstract.md", "01_intro.md", "02_model.md",
    "03_analytical.md", "04_methods.md", "05_nogo.md", "06_results.md",
    "07_selfcorrection.md", "08_discussion.md", "90_availability.md",
    "99_references.md",
]

FIGURES = """
---

## Figure and table list

All figures are produced by committed scripts from committed raw data; each can be
regenerated from a named commit (Section 9.1).

| #        | subject                          | file                                 | script                               | § |
|----------|----------------------------------|--------------------------------------|--------------------------------------|---|
| Fig. 1   | plateau A(N)                     | `d2ext_scaling.png`, `ds2_cross.png` | `analysis_ext.py`, `analysis_ds2.py` | 6.1 |
| Fig. 2   | A_plateau(k_f), stiffness memory | `s1runs_kf.png`                      | `plot_s1runs_kf.py`                  | 6.2 |
| Fig. 3   | anisotropy map A(α)              | `ds3_aniso.png`                      | `analysis_ds3.py`                    | 6.3 |
| Fig. 4   | triangle vs cosine (isotropized) | `ds3_iso.png`                        | `analysis_ds3.py`                    | 6.4 |
| Table 1  | CHSH revision: withdrawn vs valid | inline                              | —                                    | 7.2 |
"""


def strip_service(body, name):
    """Снять всё служебное: owner-строки, вето-маркеры, статусы, внутренние пути."""
    # owner-строки
    body = re.sub(r'^\*\*Section owner:\*\*.*\n\n?', '', body, flags=re.M)

    # §3.1: маркер согласования из заголовка + влить амендмент в текст
    body = body.replace(
        '## 3.1 The ontological picture  [approved: architect, deferred-veto: author]',
        '## 3.1 The ontological picture')
    body = re.sub(
        r'\n\n\*\[Amendment to the approved paragraph, per external review: the following\n'
        r'replaces the original claim that binarity and spinority "descend for\n'
        r'free"\.\]\* ',
        ' ', body)

    # вето-маркеры автора
    body = re.sub(r'\n?\[author veto point:[^\]]*\]\n\n', '\n', body)

    # служебная пометка секции
    body = body.replace('# 9. Code, Data & AI methodology (service section)',
                        '# 9. Code, Data & AI methodology')

    # References: шапка со статусами + сами маркеры
    if name == '99_references.md':
        body = re.sub(r'\*Status markers are for the author.*?\n\n', '', body, flags=re.S)
        # Всё от статус-маркера до конца записи — рабочая аннотация исполнителя
        # (статус проверки, трассировка «cited §X», заметки о недоступности источника).
        # В списке литературы остаются только библиографические данные.
        body = re.sub(r'\s*\*\*\[(?:verified|UNVERIFIED)[^\]]*\]\*\*.*?(?=\n\n\[|\n*\Z)',
                      '', body, flags=re.S)

    # внутренние файловые пути → Supplementary Information
    body = body.replace('is given in `SI_methodology.md`.',
                        f'is given in the Supplementary Information (repository: {REPO}).')
    body = body.replace('are given in `SI/SI_program_history.md`.',
                        f'are given in the Supplementary Information (repository: {REPO}).')
    body = body.replace('`seedaudit_report.md`, commit a26f76b',
                        'seed-audit report, commit a26f76b')
    body = body.replace('is catalogued in [`MODEL_FACTS.md`](MODEL_FACTS.md).',
                        f'is catalogued in the repository ({REPO}).')

    # пути картинок: sections/ → уровень paper/
    body = body.replace('](../../sim/', '](../sim/')
    return body


def words(text):
    t = re.sub(r"`[^`]*`", " ", text)
    t = re.sub(r"^\s*\|.*$", " ", t, flags=re.M)
    t = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", t)
    return len(re.findall(r"[A-Za-zА-Яа-яЁё0-9−–—]+", t))


def main():
    parts, counts = [], []
    for name in ORDER:
        with open(os.path.join(SECT, name), encoding="utf-8") as f:
            body = strip_service(f.read().rstrip() + "\n", name)
        parts.append(body)
        parts.append("\n---\n\n")
        counts.append((name, words(body)))
    parts.append(FIGURES)
    text = "".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"{'секция':<24} {'слов':>6}")
    for n, c in counts:
        print(f"  {n:<22} {c:>6}")
    print(f"  {'ИТОГО (тело)':<22} {sum(c for _, c in counts):>6}")
    print(f"\n→ {OUT}")

    service = {
        "Section owner": len(re.findall(r"Section owner", text)),
        "[author veto point": len(re.findall(r"\[author veto point", text)),
        "deferred-veto": len(re.findall(r"deferred-veto", text)),
        "Amendment to the approved": len(re.findall(r"Amendment to the approved", text)),
        "(service section)": len(re.findall(r"service section", text)),
        "[verified: / [UNVERIFIED": len(re.findall(r"\[verified:|\[UNVERIFIED", text)),
        "[*]": len(re.findall(r"\[\*\]", text)),
        "[STUB] / TODO": len(re.findall(r"\[STUB\]|TODO", text)),
        "внутренние пути (SI_/MODEL_FACTS)": len(re.findall(r"SI_methodology|SI/SI_|MODEL_FACTS", text)),
    }
    print("\nСлужебные маркеры в FINAL (все должны быть 0):")
    bad = 0
    for k, v in service.items():
        print(f"  {k:<36} {v}" + ("" if v == 0 else "   ← ОСТАЛОСЬ"))
        bad += v
    print(f"\n  {'ЧИСТО ✓' if bad == 0 else f'НЕ ВЫЧИЩЕНО: {bad}'}")


if __name__ == "__main__":
    main()
