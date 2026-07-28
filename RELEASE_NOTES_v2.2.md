# Release v2.2 — "Submission snapshot"

**Version DOI: to be minted** (Zenodo mints it when this release is published).
Concept DOI `10.5281/zenodo.21383667` — all versions, resolves to the latest.

The snapshot the journal submission of paper 3 points at. No new physics since
v2.1: this release freezes the state that the submitted PDF cites, so that the
manuscript's "Data and code availability" section and the archive agree.

## What's new since v2.1

- **Paper 3 finalized for submission.**
  - New **Data and code availability** section before References: public
    repository, the frozen release, MIT license, the bitwise-reproduction
    statement, and the pre-registration → raw → analysis order (`e87d7a9`,
    `ebade60`).
  - **Title block cleaned in the markdown source** — one title with its
    subtitle plus the author block (Artem Oktiabrev / Independent researcher,
    Ukraine / ORCID / e-mail), matching `paper/arxiv_submit/c3_submission.tex`.
    The service paragraph ("Cycle 3 draft v1 … assembly draft …") is gone: the
    claim-discipline it stated already lives in §2, and the provenance it
    promised is now the availability section (`e87d7a9`).
  - `paper/build_pdf.sh`: an empty title argument suppresses `\maketitle`, so
    the C3SUBMIT target no longer prints a second, truncated title over the
    one carried by the source (`e87d7a9`).
- **DOI loop broken.** The manuscript no longer cites the *version* DOI of the
  previous release. It names the release (v2.2) and the concept DOI, which
  resolves to whatever the latest archived version is — so a snapshot can never
  again cite a DOI minted before it existed (`ebade60`).
- **arXiv kit — variant (a), host-independent fonts.** The seven DejaVu TrueType
  files ship next to the source and are referenced by `Path=./`, so the build
  does not depend on the host's font install; control build in a clean directory
  passed with 0 errors and 0 missing characters (`a99bfe0`, `94c9fda`).
- **Hygiene.** Publication correspondence and local upload kits are out of the
  public tree (`5429dc3`, `cc62fdb`); the arXiv title block no longer leaks
  front-matter (`ecd6c28`).

## Assets

- `c3_SCIPOST_submission_v3.pdf` — **new**: paper 3 as submitted (8 pp; built
  from `paper/C3_paper_DRAFT_v1.md` at `ebade60`; the page footer carries that
  commit). Identical bytes to `paper/pdf/c3_draft_v3.pdf` in this tag;
  sha256 `26ec56dc747edefe5fec4161b25e642bbc011b07b429ad982e00221eecc5f40c`.
- `c3_draft_v3.pdf` — the same paper under its in-repo name.
- `c2_synthesis_TR.pdf` — cycle-2 synthesis, technical report.
- `main.pdf`, `si.pdf` — paper 1 (camera-ready) + supplement.

## Reproducibility

Every quantitative claim carries a commit hash; batteries are single numpy (or
JAX/GPU) scripts run under pre-registered kill criteria; the cycle reproduces
bitwise. Repository: <https://github.com/aoktyabrev/ribbon-at-the-bell-bound>.

---

## Release commands (for the operator)

The tag must sit on the submission commit chain — **`3cfeb99`**, the rebuilt PDF,
whose source commit `ebade60` is what the PDF footer prints. Nothing may be
committed on top before tagging.

```bash
cd /home/artem/quantum_entanglement

# 0. sanity: HEAD must be 3cfeb99 and the tree clean
git log --oneline -1          # → 3cfeb99 paper3: пересборка submission-PDF …
git status --short            # → empty

# 1. annotated tag, strictly on the submission commit
git tag -a v2.2 3cfeb99 -m "v2.2 — submission snapshot (paper 3 as submitted)"
git push origin v2.2
# (if ssh fails with "Permission denied (publickey)", the HTTPS route works:
#  git -c credential.helper='!gh auth git-credential' \
#      push https://github.com/aoktyabrev/ribbon-at-the-bell-bound.git v2.2 )

# 2. GitHub Release with the five assets
gh release create v2.2 \
  --title "v2.2 — Submission snapshot" \
  --notes-file RELEASE_NOTES_v2.2.md \
  private_correspondence/c3_SCIPOST_submission_v3.pdf \
  paper/pdf/c3_draft_v3.pdf \
  paper/pdf/c2_synthesis_TR.pdf \
  paper/pdf/main.pdf \
  paper/pdf/si.pdf
```

Note on the first asset: `private_correspondence/` is gitignored, but
`gh release` uploads files from disk, not from the tree — the path works. It is
byte-identical to `paper/pdf/c3_draft_v3.pdf`; upload it under the submission
name only if you want the asset list to say plainly which file went to the
journal, otherwise drop that line.

Zenodo mints the version DOI once the GitHub Release is published (the webhook
must be enabled for this repository). After the DOI exists, the follow-up pass
updates: this file's header, `RELEASES.md`, `README.md` ("Latest release" line),
and the data-availability lines in `private_correspondence/scipost_form_fields.txt`
and `cover_letter_SCIPOST.md`. The manuscript itself needs **no** edit — it
cites the concept DOI on purpose.
