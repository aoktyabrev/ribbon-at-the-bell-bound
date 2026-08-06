"""Sync check across the five hand-maintained sources of paper 3.

There is no generator: the markdown and the four .tex files are edited
separately, so a patch applied to some of them and not the others leaves no
trace. This script is the missing trace.

Two layers:

  L1  anchor phrases -- a registry of the claims that reviewers and kinship
      checks act on. Each anchor must occur the SAME number of times in every
      source. Exact, no heuristics, and it is the layer that actually gates.

  L2  normalized body diff -- both formats are folded to a token stream
      (LaTeX markup expanded, unicode folded, case dropped) and compared
      against the markdown. Heuristic by nature, so it reports rather than
      gates, and it catches drift that no anchor anticipated.

Exit code: 0 if L1 clean, 1 otherwise.  L2 findings never change the exit
code -- see REPORT_ONLY below.

Usage:  python paper/check_sync.py [-v]
"""

from __future__ import annotations

import difflib
import pathlib
import re
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

SOURCES = [
    ROOT / "paper" / "C3_paper_DRAFT_v1.md",
    ROOT / "paper" / "arxiv_submit" / "c3_submission.tex",
    ROOT / "paper" / "arxiv_submit" / "c3_fop_source.tex",
    ROOT / "paper" / "arxiv_submit" / "c3_fop_source_pdflatex.tex",
    ROOT / "private_upload" / "arxiv_paper3" / "c3_submission.tex",
]
REFERENCE = SOURCES[0]

REPORT_ONLY = True          # L2 reports; only L1 sets the exit code

# ---------------------------------------------------------------------------
# L0 -- the release label, across BOTH papers
# ---------------------------------------------------------------------------
# Paper 1 has a generator and a single source of truth, so it needs no L1/L2.
# It still shares one fact with paper 3: the release its availability section
# names. That fact lives in two unrelated pipelines, and nothing textual links
# them -- bump paper 3's five sources, forget paper 1, and both papers build
# clean while naming different snapshots. The merge that produced v2.3 already
# demonstrated the failure mode inside paper 3; this closes the same hole
# across the pair.
#
# Bump RELEASE on every release, together with README and RELEASES.md.
RELEASE = "release v2.3"

RELEASE_FILES = SOURCES + [
    ROOT / "paper" / "sections" / "90_availability.md",   # paper 1, source
    ROOT / "paper" / "FINAL_v1.md",                       # paper 1, generated
]


# ---------------------------------------------------------------------------
# L1 -- anchor registry
# ---------------------------------------------------------------------------
# Each anchor is matched against the NORMALIZED text, so it is written in
# normalized form: lowercase, unicode folded, whitespace collapsed.
#
# Add an anchor whenever a claim becomes load-bearing -- typically when it is
# patched, cited in a kinship check, or raised in review. An anchor that is
# deliberately being removed stays here until it is removed from all five.
ANCHORS = [
    # --- П1: Svetlichny / genuine multipartite (GWAN12) ---
    "genuine n-partite nonlocality",
    "at the svetlichny bound",
    "violates the svetlichny inequality",
    # --- П2: N-partite steering cut (SBCSV15) ---
    "registered next open problem",
    "no-signaling plus steering structure",
    "post-quantum steering",
    # --- П3: k-producibility attribution (BBGP09/BBGP13/CLG15) ---
    "2-producible ceiling",
    "k-producible",
    # --- П4: causal separability (OCB12) ---
    "shared order-coin",
    "causally separable",
    # --- structural claims that must not drift silently ---
    "statistically invisible precedence structure",
    "schedule-invisibility",
    "algebraic (4.0 at n = 4)",
    # --- data availability: found missing from c3_submission.tex by L2 on the
    #     first run of this script, present in the markdown and both FoP kits ---
    "all code, pre-registrations, raw measurements",
    "zenodo",
    "mit license",
    # --- AI disclosure: the role terms are load-bearing (one term per party,
    #     no bare "operator"/"architect" in article text) and the section is
    #     hand-inserted into every source, so it is exactly the kind of patch
    #     that drifts. Reproduction attribution is anchored for the same reason.
    "under an explicit division of roles",
    "an ai architect role",
    "an ai executor role (claude code)",
    "accepts full responsibility for the manuscript",
    "reproduced by the author in a separate environment",
    # --- figure captions (§5). These are the reason "figure" is not stripped:
    #     the protocol line of each trilemma edge is what stops the DS3 and
    #     ISO-DYN edges from reading as a contradiction, and the operator note
    #     is what stops the CHSH 4.0 from being read as the N=4 Mermin 4.0.
    "the chsh landscape",
    "the protocol of each edge named",
    "applies the haar rotation post hoc",
    "entering the relaxation as clamps",
    "not a symmetry of the model",
    "brackets the 1/3-cosine point",
    "belong to different operators",
    # --- the release label. git merges the two sections around it cleanly and
    #     still leaves one source saying v2.2 and the rest v2.3: a semantic
    #     conflict no textual merge can see. Bump on every release.
    "frozen as release v2.3",
]


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
MACROS = {
    r"\cdot": ".", r"\times": "x", r"\to": "->", r"\leftrightarrow": "<->",
    r"\Leftrightarrow": "<=>", r"\Rightarrow": "=>", r"\rightarrow": "->",
    r"\ge": ">=", r"\le": "<=", r"\neq": "!=", r"\pm": "+-",
    r"\perp": "perp", r"\otimes": "ox", r"\approx": "~",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\theta": "θ", r"\lambda": "λ",
    r"\mu": "μ", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Delta": "Δ", r"\Sigma": "Σ", r"\Omega": "Ω", r"\Lambda": "Λ",
    r"\textgreater": ">", r"\textless": "<", r"\textbar": "|",
    r"\ldots": "...", r"\dots": "...", r"\sqrt": "sqrt",
}

UNICODE_FOLD = {
    "—": "-", "–": "-", "−": "-", "‐": "-", "‑": "-",
    "≥": ">=", "≤": "<=", "≠": "!=", "±": "+-",
    "√": "sqrt", "·": ".", "×": "x", "⊗": "ox", "⊥": "perp", "≈": "~",
    "→": "->", "↔": "<->", "⇔": "<=>", "⇒": "=>",
    "½": "1/2", "⅓": "1/3", "¼": "1/4",
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5",
    "“": '"', "”": '"', "‘": "'", "’": "'", "„": '"',
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
}

# "figure" is deliberately NOT stripped: the paper-3 figure captions carry the
# protocol of each trilemma edge, which is exactly the text that must not drift.
STRIP_ENVS = ("tabular", "longtable", "table", "thebibliography")


def _strip_tex(text: str) -> str:
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.S)
    if m:
        text = m.group(1)

    text = re.sub(r"(?<!\\)%.*", "", text)                  # comments
    for env in STRIP_ENVS:
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", text, flags=re.S)

    text = re.sub(r"\\(rule|label|hypertarget|phantomsection|maketitle|newpage"
                  r"|includegraphics"
                  r"|clearpage|thispagestyle|pagestyle|fancyhf|cfoot|rfoot"
                  r"|lfoot|setcounter|vspace|hspace|bibliography|bibliographystyle)"
                  r"\s*(\[[^\]]*\])?(\{[^{}]*\})*", " ", text)
    text = re.sub(r"\\(sub)*section\*?\s*\{[^{}]*\}", " ", text)

    # math delimiters: keep the content, drop the markers
    text = text.replace("$$", "").replace("$", "")

    for macro, repl in sorted(MACROS.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(macro, repl)

    # accents: \'i -> i, \"o -> o, \v{s} -> s, \c{c} -> c
    text = re.sub(r"\\[\'\"`^~=.]\{?([A-Za-z])\}?", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{([A-Za-z])\}", r"\1", text)

    # text-level wrappers: keep the argument
    for _ in range(6):
        text, n = re.subn(
            r"\\(textbf|textit|emph|texttt|textsc|textrm|mathrm|mbox|text"
            r"|underline|sout|st|uline|footnote)\s*\{([^{}]*)\}", r"\2", text)
        if not n:
            break

    text = re.sub(r"\\begin\{[a-zA-Z*]+\}|\\end\{[a-zA-Z*]+\}", " ", text)
    text = re.sub(r"\\item\b", " ", text)
    text = text.replace(r"\^{}", "^").replace(r"\\", " ")
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)              # leftover commands
    text = text.replace("{[}", "[").replace("{]}", "]")
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\\([&%_#$ ])", r"\1", text)
    text = text.replace("~", " ")
    return text


def _strip_md(text: str) -> str:
    # The .tex sources carry title and author in the preamble, which is dropped
    # with everything before \begin{document}; drop the same matter here.
    m = re.search(r"^##\s*Abstract\s*$", text, re.M)
    if m:
        text = text[m.start():]

    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):                    # tables
            continue
        if re.fullmatch(r"-{3,}|={3,}", s):      # rules
            continue
        if re.match(r"^#{1,6}\s", s):            # headings: dropped in tex too
            continue
        s = re.sub(r"^>\s?", "", s)              # blockquote
        s = re.sub(r"^[-*+]\s+", "", s)          # bullets
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"`{1,3}", " ", text)
    text = re.sub(r"\*{1,3}", " ", text)
    # images: dropped whole, alt text included -- the tex side has no alt text,
    # and \includegraphics is dropped there, so this keeps the two symmetric.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)     # links
    return text


def normalize(path: pathlib.Path) -> str:
    raw = path.read_text(encoding="utf-8")
    text = _strip_tex(raw) if path.suffix == ".tex" else _strip_md(raw)

    text = text.replace("``", '"').replace("''", '"')        # TeX quotes
    text = text.replace("---", "-").replace("--", "-")
    for src, dst in UNICODE_FOLD.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)          # space before punct
    text = re.sub(r"([(\[|])\s+", r"\1", text)               # space after opener
    return text.strip()


def tokens(norm: str) -> list[str]:
    return [t for t in re.split(r"\s+", norm) if t]


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------
def layer1(normed: dict[pathlib.Path, str]) -> list[str]:
    failures = []
    print("L1  anchor phrases (must match across all five sources)")
    width = max(len(a) for a in ANCHORS)
    for anchor in ANCHORS:
        counts = {p: normed[p].count(anchor) for p in SOURCES}
        vals = set(counts.values())
        ok = len(vals) == 1
        flag = "ok " if ok else "MISMATCH"
        shown = "/".join(str(counts[p]) for p in SOURCES)
        note = "" if any(counts.values()) else "   (absent everywhere)"
        print(f"  {flag}  {anchor:<{width}}  {shown}{note}")
        if not ok:
            failures.append(anchor)
            for p in SOURCES:
                print(f"          {counts[p]}  {p.relative_to(ROOT)}")
    return failures


def layer2(normed: dict[pathlib.Path, str], verbose: bool) -> int:
    print("\nL2  normalized body diff against "
          f"{REFERENCE.relative_to(ROOT)}  (report only)")
    ref = tokens(normed[REFERENCE])
    total = 0
    for path in SOURCES[1:]:
        cur = tokens(normed[path])
        sm = difflib.SequenceMatcher(a=ref, b=cur, autojunk=False)
        hunks = [op for op in sm.get_opcodes() if op[0] != "equal"]
        ratio = sm.ratio()
        total += len(hunks)
        print(f"  {path.relative_to(ROOT)}: similarity {ratio:.4f}, "
              f"{len(hunks)} differing region(s)")
        if verbose:
            for tag, i1, i2, j1, j2 in hunks[:40]:
                a = " ".join(ref[i1:i2])[:110]
                b = " ".join(cur[j1:j2])[:110]
                print(f"      {tag:<7} ref[{i1}:{i2}] {a!r}")
                print(f"      {'':<7} cur[{j1}:{j2}] {b!r}")
    return total


def layer0() -> list[pathlib.Path]:
    """The release label, in both papers. Raw text: no normalization needed,
    and none wanted -- the point is the literal string a reader will see."""
    print(f"L0  release label {RELEASE!r} (both papers)")
    width = max(len(str(p.relative_to(ROOT))) for p in RELEASE_FILES)
    bad = []
    for p in RELEASE_FILES:
        n = p.read_text(encoding="utf-8").count(RELEASE) if p.exists() else -1
        flag = "ok  " if n >= 1 else "MISS"
        if n < 1:
            bad.append(p)
        print(f"  {flag} {str(p.relative_to(ROOT)):<{width}}  {n}")
    return bad


def main() -> int:
    verbose = "-v" in sys.argv
    missing = [p for p in SOURCES if not p.exists()]
    if missing:
        for p in missing:
            print(f"MISSING: {p}")
        return 1

    stale = layer0()
    print()

    normed = {p: normalize(p) for p in SOURCES}
    failures = layer1(normed)
    layer2(normed, verbose)

    print()
    if stale:
        print(f"FAIL: {RELEASE!r} missing from "
              + ", ".join(str(p.relative_to(ROOT)) for p in stale))
    if failures:
        print(f"FAIL: {len(failures)} anchor(s) out of sync: "
              + ", ".join(repr(a) for a in failures))
    if stale or failures:
        return 1
    print("PASS: all anchors consistent across the five sources, "
          "release label consistent across both papers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
