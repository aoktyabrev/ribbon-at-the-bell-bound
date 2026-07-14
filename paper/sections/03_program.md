# 3. Simulation program  [STUB]

**Section owner:** Joint (Architect design + Code execution)

## Theses (from outline §3)
- Phases A–D: A scaffold+R0; B gauge-blindness; C entropic singlet E=−tanh(ΔS cosθ/2);
  D topology in R⁴ (ℤ₂-framing).
- Methodology: pre-registration (*_prereg.md), kill-criteria, mirror pairs (a,b)↔(−a,−b),
  honest sampler, ordering "raw → commit → analysis".
- Two-sided audit: "too-good" (CHSH>2) = STOP+audit trigger, not celebration.
- Test coverage BY INVARIANT (not by count): Bishop SO(4) (R t=t', RᵀR=I, det=+1,
  minimality), ℤ₂-parity (T-inv-1/2/3), mirror symmetry, singularity detection
  (temporal/tangent/spatial lift-wall).

## Sources
- ribbon_model_note.md; phase_C_report.md; D0–DS3 reports; all *_prereg.md.
- Code: band4d.py, measurement.py, test_bishop.py, test_invariant.py.
- Commit ordering: every phase-D "…-raw" (frozen) → "…-analysis" pair.
