# Ribbon model — paper outline

Structure + 3–5 theses per section, with references to reports/commits. No section prose.
Refs: reports = sim/phase_D/*_report.md, ribbon_model_note.md; commits = short hashes.
Language of everything under paper/ is English from here on.

## 1. Introduction
- Ontology: "particles = boundaries of an extended object (the ribbon) embedded in a
  higher dimension"; binary and spinorial outcomes emerge from geometry, not postulate
  (ribbon_model_note §"origin of spinority from embedding").
- Maximalist bet: derive the Born rule (isotropic cosine, A=1, CHSH=2√2) from a classical
  elastic ribbon with an honest basin measure.
- Honest framing: pre-registration of hypotheses + kill-criteria BEFORE runs; the bet's
  fate is recorded, not fitted (methodology §3).
- Result headline: the bet is NOT won; instead we map the ribbon's POSITION WITHIN THE
  CLASSICAL REGION — the family's limit point is the Bell bound (CHSH→2), not the quantum
  bound (2√2) — and name the structural locks (U(1)-theorem, trilemma). (consistent with §5)

### 1a. Related work
- Bell / CHSH: local-realistic bound framework; the "saw" (triangle) as the shared-λ
  local model's signature.
- Gisin 1990 (Born rule from no-signalling): INDEPENDENTLY rediscovered here; provenance
  by commits (born.py/born2.py, session 3; Δ numbers).
- Pearle 1970 (detection loophole): directly relevant to self-correction audit #2 (DS3).
- Contrast programs — de Broglie–Bohm (nonlocal hidden variables), 't Hooft cellular
  automata, Palmer Invariant Set Theory: differentiation — our genre is SYSTEMATIC
  MAPPING of a SINGLE ontology under pre-registration (an exhaustive falsification map),
  not a new interpretation.

## 2. Theory
- Chord law: uniqueness p=2 from causality, not from single-pair consistency
  (ribbon_model_note §2.4–2.6; born.py, chord.py, session 3).
- Born from causality (rediscovery of Gisin 1990): deviation from the square in a
  one-parameter family ⇒ superluminal telegraph; Δ=ε·|(1−2q)³−(1−2q)|
  (ribbon_model_note §2.5; born2.py, q=0.85, Δ=0.0717).
- "Conservation of the postulate": the square is not derivable from consistency
  (counterexample), but is fixed by causality — the postulate relocates, it does not
  vanish (§2.4 + §2.5 + §2.7 "postulate-conservation").
- [The dimensional/area hypothesis is DEFERRED to §7 as a [conjecture], next to open
  problem E.]

## 3. Simulation program
- Phases A–D: A scaffold+R0; B gauge-blindness; C entropic singlet E=−tanh(ΔS cosθ/2);
  D topology in R⁴ (ℤ₂-framing) (ribbon_model_note; phase_C_report; D0–DS3 reports).
- Methodology: pre-registration (*_prereg.md), kill-criteria, mirror pairs (a,b)↔(−a,−b),
  honest sampler, ordering "raw → commit → analysis" (all phase-D commits).
- Two-sided audit: a "too-good" result (CHSH>2) is a STOP+audit trigger, not a celebration
  (DS2, DS3 audits).
- Test coverage by INVARIANT (not by test count): Bishop SO(4) transport (R t=t', RᵀR=I,
  det=+1, minimality), ℤ₂-parity (T-inv-1/2/3), mirror symmetry, singularity detection
  (temporal/tangent/spatial lift-wall) (test_bishop.py, test_invariant.py; band4d, measurement).

## 4. No-go results
- Gauge-blindness (R5): any axial observable factorizes through SU(2)→SO(3), blind to the
  ℤ₂ fiber (ribbon_model_note §"no-go gauge-blindness").
- U(1)-theorem (axis ≠ frame): the ℤ₂ class lives in the U(1) fiber (boundary lift sign);
  ℤ₂⊂ker(S³→SO(3)) inaccessible to axial measurement; "blind slice" is a theorem, not a
  defect (D1_report §3–5; lift_twin: max|Δn|=0, parity flip).
- Sector-blindness of the axial signal: even≈odd under uniform preparation (S1-runs R3:
  ΔA~1σ); axial correlation carries no topology (S1runs_report §R3).
- Census D1-A: all 16 cells EXIST ⇒ no geometric prohibition (no null measure)
  (census_notes.md, D1_report §2).

## 5. Quantitative boundary
- Kinetic plateau (non-dilutability): A∞=0.363±0.012, M0≫M1 ΔAICc=6.36 (D2-ext); holds at
  k_f×4 (drop16→96=+0.002±0.023, DS2) — basin mechanism, not finite-size. BOUNDED EVIDENCE:
  N∈[16,96], k_f∈{×1,×4} (D2ext_report, DS2_report §1).
- Amplitude origin = stiffness memory: A(k_f)=0.27/0.42/0.56/0.84, athermal, sector-blind
  (S1runs_report; "entropic → kinetic" refinement).
- Anisotropy map: A(α)~cosα (k_f×1) → ~cos²α (k_f×4); axis ê privileged
  (DS3_report §1; ds3_aniso.png).
- Trilemma amplitude–form–isotropy: cosine only anisotropically; honest isotropization ⇒
  triangular LHV E=−ρ(1−2θ/π), CHSH=2ρ; ρ(k_f)=0.374/0.810 (DS3_report §3; ds3_iso.png;
  ribbon_model_note §"trilemma").
- Family limit = Bell bound: k_f→∞ ⇒ ρ→1, CHSH→2; position within classical region,
  distance to optimal LHV = 2−CHSH = 0.38 at k_f×4 (DS3_report §4).

## 6. Case studies in self-correction
- Audit #1 (DS2): isotropic CHSH formula on an anisotropic system gave a spurious S=2.58;
  direct two-end measurement ⇒ E(π/2,·)≈0, isotropy false, direct |S|=1.21
  (DS2_report §4; audit_ds2_chsh.py).
- Audit #2 (DS3): S=2.39 from setting-dependent DEGENERATE post-selection — detection
  loophole (Pearle 1970); without post-selection |S|=1.62 (DS3_report §2; analysis_ds3 fix).
- Meta-thesis: the pre-registered kill "S>2 → stop-audit" caught a METHOD artifact (not
  physics) twice; no Bell violation survived audit.
- Program lesson: a "too-good" result in a classical model = a protocol-bug signal (CHSH
  canon line revised; retracted numbers kept struck-through, not deleted).

## 7. Discussion
- Final phase-D lemma: realized {binarity, spinority, anticorrelation, non-dilutability,
  anisotropic cosine}; not realized {exact zeros, isotropic cosine, CHSH>2}
  (ribbon_model_note §"final lemma").
- Open problem E: does a physical fiber-breaking invariant (second fundamental form /
  privileged slice normal) exist, giving BOTH an empty aligned basin AND an isotropic
  cosine; equivalent to escaping the trilemma (ribbon_model_note §"open problem E").
- [conjecture] Area hypothesis (moved from §2): basins ∝|chord|¹ → saw (classical),
  ∝|chord|² → Born; "the ribbon has a face" (ribbon_model_note §2.7 "dimensional hook",
  §3 open task μ(B)=|s·a−t·b|²/8) — states next to open problem E as the untested route.
- Connection to theorems: Bell (the saw/triangle is the shared-λ local model's bound — the
  E(45°)=−0.502 value in ribbon_model_note §2.7 is the TRIANGULAR-LHV value vs quantum
  −0.707, i.e. a Bell result); Kohen–Specker (1967) role = EXISTENCE OF A MEASURE /
  RELOCATION OF THE POSTULATE (single-end construction yields cos²(θ/2) honestly, but the
  postulate moves into the microstate measure — §2.7 "existence", §2.4 non-derivability);
  Gisin 1990 (Born from signalling). The ribbon is a constructive realization of these bounds.
- Limitations: classical (not quantum) dynamics; ONE class of discretizations (framed curve
  in R⁴, axial readout); U(1)-theorem holds for axial datums; bounded N and k_f ranges.

## 8. Code, Data & AI methodology (service section)
- Code & data availability: repository (sim/phase_D/), frozen raw in *_raw*.json (committed
  BEFORE analysis), full provenance by commit hashes (raw → analysis pairs per stage).
- Reproducibility: CPU-JAX float64, fixed PRNGKeys, pre-registration files (*_prereg.md +
  addenda) fixed before runs.
- AI methodology: Claude (architect/reviewer) designs and pre-registers; Claude Code
  (executor) implements and runs; the human owner holds decisions and veto. Pre-registration
  + kill-criteria + two-sided audit serve as controls over BOTH AI roles.

## Figure list (draft)
1. A(N) plateau (D2-ext + DS2 cross-scan): d2ext_scaling.png, ds2_cross.png.
2. A(α) anisotropy map: ds3_aniso.png.
3. Triangle vs cosine (isotropization): ds3_iso.png.
4. CHSH revision table (retracted/valid): from ribbon_model_note §"CHSH revision".
5. A(k_f) stiffness memory (S1-runs): to be produced from S1runs results.
(Optional) Phase-C singlet E(θ).
