# The Ribbon at the Bell Bound

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21383667.svg)](https://doi.org/10.5281/zenodo.21383667)
*(concept DOI — all versions)*

Pre-registered simulations mapping the classical boundary of a geometric entanglement model —
code, data, and provenance for *"The Ribbon at the Bell Bound"*.

## What this is

This repository holds the complete record of a simulation program that took a classical,
geometrically motivated model of entanglement and drove it to its quantitative limit. The
model represents an entangled pair as one framed curve embedded in R⁴, its two ends being the
observed particles; the framing carries a natural ℤ₂ topological label, a candidate origin for
outcome binarity and spinor structure.

The program's maximal stake — that classical relaxation dynamics of this object reproduces
singlet statistics — was lost, and the paper reports exactly how. What survives is a mapped
boundary: a structural theorem showing the ℤ₂ label is invisible to axial readout, a census
finding no forbidden outcome branch, a length-stable but stiffness-controlled and anisotropic
correlation amplitude, and two apparent Bell violations that both died in pre-registered
audits. The negative result, quantified and named, is the contribution.

**Paper:** [arXiv link TODO after submission]. Camera-ready:
[`paper/FINAL_v1.md`](paper/FINAL_v1.md) / [`paper/pdf/main.pdf`](paper/pdf/main.pdf);
supplementary: [`paper/SI_v3.md`](paper/SI_v3.md) / [`paper/pdf/si.pdf`](paper/pdf/si.pdf).
The working draft [`paper/DRAFT_v3.md`](paper/DRAFT_v3.md) is retained as a documented stage.

**Archived snapshot:** https://doi.org/10.5281/zenodo.21383967 (release `paper-v1.1`,
camera-ready state). The concept DOI https://doi.org/10.5281/zenodo.21383667 resolves to the
latest archived version. An earlier snapshot (`paper-v1`, pre-camera-ready) is retained as
version 1 of the same record.

To cite the archive *(version DOI — camera-ready snapshot)*:

> Oktiabrev, Artem. *The Ribbon at the Bell Bound: Mapping the Classical Boundary of a
> Geometric Entanglement Ontology.* Zenodo, 2026. https://doi.org/10.5281/zenodo.21383967

## Reproducibility

Two properties are kept separate, because they are different claims.

**Repeatability is exact.** Every run script fixes its PRNGKey seed protocol, so the same seed
returns the same number. Every figure in the paper regenerates from committed raw data by
running its committed script:

| Figure | Subject | Script | Raw data frozen at |
|---|---|---|---|
| Fig. 1a | Plateau A(N), D2-ext scaling | `sim/phase_D/analysis_ext.py` | `624b628` (`phase-D2ext-raw`) |
| Fig. 1b | Cross-scan at two stiffnesses | `sim/phase_D/analysis_ds2.py` | `a9f6cfe` (`phase-DS2-raw`) |
| Fig. 2 | A_plateau(k_f), stiffness memory | `sim/phase_D/plot_s1runs_kf.py` | `704e29c` (`phase-D-S1runs-raw`) |
| Fig. 3 | Anisotropy map A(α) | `sim/phase_D/analysis_ds3.py` | `3d8dd67` (`phase-DS3-raw`) |
| Fig. 4 | Triangle vs cosine, isotropized | `sim/phase_D/analysis_ds3.py` | `3d8dd67` (`phase-DS3-raw`) |
| Table 1 | CHSH revision | — (inline) | canon at `311d3ae` |

Figures land in `sim/phase_D/fig/`. The seed audit's raw data is frozen at `ecff715`, its
analysis at `a26f76b`.

**Reproducibility under a change of seed was measured, not assumed.** A pre-registered audit
re-ran the N = 32 and N = 96 cells under twelve fresh base keys; the cross-seed scatter came
out consistent with the quoted binomial errors (r = 0.88 over eleven repeats of the core cell,
r = 1.11 at N = 96). See `sim/phase_D/seedaudit_report.md`.

**Pre-registration discipline.** Every campaign has a `*_prereg.md` fixing hypotheses,
quantitative expectations, and kill-criteria. What the commit record proves is graded, and the
paper states it that way: no pre-registration was ever edited after entering the record (eight
files, one commit each); every one entered the record before the analysis it governs (a
`*-raw` commit precedes each `*-analysis` commit); and for the final campaign, the seed audit,
the pre-registration was committed before the runs themselves (`2b946ae` → `ecff715` →
`a26f76b`). The working arrangement behind this — including the errors it caught, in both
directions between the two AI roles — is documented in
[`paper/SI/SI_methodology.md`](paper/SI/SI_methodology.md).

The model as implemented, with the places where the code differs from the tidy version of the
mathematics, is catalogued in [`MODEL_FACTS.md`](MODEL_FACTS.md).

## Repository map

```
paper/        Manuscript: sections/, SI/, pdf/, builders, FINAL_v1.md, DRAFT_v3.md,
              SI_v3.md, REVIEW_1.md, REVIEW_2.md
sim/          Simulation code and data
  src/ribbon_sim/   Phase A–C toolkit (quaternion frames, energy, dynamics)
  phase_D/          Phase D model, campaigns, pre-registrations, reports, figures
    results/        Frozen raw measurements (*.json), committed before analysis
  tests/            Phase A–C test suite
  experiments/      Phase A–C run configs
  results/          Phase A–C reports and decisions.md
notes/        Provenance that is not the paper: AI role contexts, historical material
MODEL_FACTS.md      The model as implemented, with file:line references
SPEC.md             The simulation specification, fixed before the first physics campaign
ribbon_model_note.md  The canonical record (analytical layer + phase results)
```

`SPEC.md` and `ribbon_model_note.md` are in Russian, as are the campaign reports under
`sim/phase_D/` and everything in `notes/`. The paper is in English.

## Requirements

JAX. Phase A–C runs use float32 and can use a GPU; **all phase-D runs are CPU-JAX in float64**
(`JAX_PLATFORMS=cpu`, `jax_enable_x64`), because parity is the sign of a quaternion lift
accumulated along the whole chain and we wanted precision headroom for it. fp64 on the
development GPU (RTX 4070 Ti) is ~1/64 speed and the cells are small, so CPU is not a
sacrifice.

```bash
cd sim && python -m venv .venv && source .venv/bin/activate
pip install "jax>=0.4.30" numpy matplotlib pyyaml pytest   # CPU-only; see note
python -m pytest tests/ phase_D/ -q                        # 60 + 9 tests
JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/plot_s1runs_kf.py
```

Note on dependencies: `sim/pyproject.toml` pins `jax[cuda12]`, which is what the development
machine used for phases A–C. Nothing in phase D — the campaigns behind every number in the
paper — needs CUDA, so plain `jax` is enough to reproduce them. The project is configured as
an application rather than an installable package (`[tool.uv] package = false`); imports work
through the src-layout path, which is why the commands above set `PYTHONPATH` instead of
installing.

Phase-D campaigns take minutes to an hour on CPU; the seed audit is ~50 min.

## License

Code is MIT ([`LICENSE`](LICENSE)).

The text of the paper — everything under `paper/`, plus `MODEL_FACTS.md`, `SPEC.md`,
`ribbon_model_note.md`, and `notes/` — is licensed separately under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
