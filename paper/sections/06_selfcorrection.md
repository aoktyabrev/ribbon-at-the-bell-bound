# 6. Case studies in self-correction  [STUB]

**Section owner:** Joint (Code audits + Architect methodology)

## Theses (from outline §6)
- Audit #1 (DS2): isotropic CHSH formula on an anisotropic system gave spurious S=2.58;
  direct two-end measurement ⇒ E(π/2,·)≈0, isotropy false, direct |S|=1.21.
- Audit #2 (DS3): S=2.39 from setting-dependent DEGENERATE post-selection — detection
  loophole (Pearle 1970); without post-selection |S|=1.62.
- Meta-thesis: pre-registered kill "S>2 → stop-audit" caught a METHOD artifact (not
  physics) twice; no Bell violation survived audit.
- Program lesson: "too-good" in a classical model = protocol-bug signal; retracted CHSH
  numbers kept struck-through, not deleted.

## Sources
- Reports: DS2_report.md §4; DS3_report.md §2.
- Code: audit_ds2_chsh.py, analysis_ds3.py (post-selection fix), DS2_audit_chsh.json.
- Canon: ribbon_model_note.md §"CHSH revision".
