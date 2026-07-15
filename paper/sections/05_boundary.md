# 5. Quantitative boundary

**Section owner:** Code

## 5.1 The plateau

The one quantity the ribbon carries that does not dilute with length is the axial readout
amplitude of the odd sector. Fit against three models — a constant A∞, a power law A·N^−γ,
and a saturating A∞ + c·N^−γ — the constant wins decisively: A∞ = 0.363 ± 0.012, with the
power law disfavoured by ΔAICc = 6.36 and its exponent fitted negative (no decay) (D2-ext;
commit 2784edf). The plateau holds at a second stiffness: cross-scanning N ∈ {16, 32, 64, 96}
at k_f×1 and k_f×4, the amplitude change from N = 16 to N = 96 is +0.010 ± 0.038 and
+0.002 ± 0.023 respectively — flat within error at both couplings (DS2; commit f928dd4). The evidence is
bounded: N ∈ [16, 96], k_f ∈ {×1, ×4}. Within that window the plateau is kinetic, not
thermodynamic — the amplitude is flat in temperature (A(T) = 0.368 / 0.397 / 0.353 at
T = 0.025 / 0.05 / 0.10, unchanged within σ ≈ 0.027) (S1-runs; commit a9cef7b) — so it is a
property of the basin structure, not of thermal fluctuation.

[Fig. 1: plateau — d2ext_scaling.png + ds2_cross.png]

## 5.2 Origin of the amplitude

The amplitude is stiffness-controlled. Sweeping the twist coupling gives
A(k_f) = 0.27 / 0.42 / 0.56 / 0.84 for k_f × {0.5, 1, 2, 4} — a strong
monotone dependence that refutes any fixed geometric-measure origin
(S1-runs R1; commit a9cef7b). The A∞ ≈ 0.36 of Section 5.1 is therefore
not a fundamental constant but the value of a dial at baseline stiffness;
the weakness of the signal is a property of soft coupling, not a limit of
the mechanism. At θ = 0 the amplitude is, by identity, A = 2·P_aligned − 1
(P_aligned = 0.68 at the baseline plateau), so explaining the amplitude
means explaining the aligned-basin weight. Two measured facts locate that
weight in kinetics rather than thermodynamics. First, it is flat in
temperature (Section 5.1), while any Boltzmann reading of the landscape
would move with T: the census finds the four branch minima spread over
0.22 in energy (branch means −3.78 for both branches, Δ ≈ 0.006), and no
assignment of Boltzmann weights to these energies reproduces a
temperature-independent 68/32 split (D1 census; D_synthesis_S1 §2).
Second, it is flat in chain length from N = 16 to N = 96 at both measured
stiffnesses (Section 5.1), while an equilibrium chain correlation with a
finite correlation length would decay. What remains is relaxation itself:
the split is set by the geometry of the attraction domains under the
clamped dynamics — a domain volume controlled by coupling stiffness. This
is the reading fixed in the canonical record after the cross-scan (commit
311d3ae): the amplitude is a stiffness-controlled kinetic correlation,
athermal and sector-blind (Section 4.3).

![Fig. 2: A∞(k_f) — stiffness memory](../../sim/phase_D/fig/s1runs_kf.png)

*Fig. 2. A∞(k_f) — stiffness memory. Four points with 1σ error bars from the frozen
S1-runs R1 raw data (N = 32, M = 1200, T = 0.05; commit a9cef7b). Dotted line: the
D2-ext plateau A∞ = 0.363 of Section 5.1, measured over N ∈ [16, 96].*

## 5.3 Anisotropy map

The cosine angular law the correlation appears to follow is anisotropic. Tilting the clamp axis
a by α from the privileged axis ê and reading the antiparallel amplitude A(α) = |E_anti|, the
signal decays smoothly to zero at α = π/2: from 0.387 to 0.005 at k_f×1 and from 0.865 to 0.008
at k_f×4 (DS3; commit 0fb5452). The best-fit law is cos α at k_f×1 and steepens to ~cos²α (an
exponential fitting equally well) at k_f×4. The axis ê — simultaneously the twist axis and the
readout axis — is strongly privileged; the cosine dependence exists only along it, which is
exactly why the isotropic CHSH estimator of Section 6 was invalid.

[Fig. 3: anisotropy map — ds3_aniso.png]

## 5.4 The trilemma and the triangle

Averaged honestly over the shared randomness λ = (R, n_A, n_B) — where R ~ Haar on SO(3) is
the ribbon's orientation, common to both ends through the geometry, while the settings n_A and
n_B are drawn independently — the correlation is not a cosine. Fitting the
isotropized E(θ) against both forms, the triangular local-realistic function E = −ρ(1 − 2θ/π)
beats the cosine by a wide margin in χ²: 8 vs 39 at k_f×1 and 1 vs 218 at k_f×4 — the data lie
on the straight line, not the curve (DS3; commit 0fb5452). This is the shared-λ local model's
signature, and its CHSH value is exactly S = 2ρ, with ρ the source-alignment amplitude:
ρ = 0.374 / 0.810 gives |S| = 0.75 / 1.62 at k_f × {1, 4}. Three properties therefore trade
off and cannot be held together: a cosine form exists only anisotropically (5.3); honest
isotropy forces the triangle (this subsection); and the amplitude ρ is bought with stiffness
(5.2). The family, swept in k_f, approaches the Bell bound from below — |S| = 1.62 at k_f×4
sits a measured distance 2 − 1.62 = 0.38 short of it — with the deterministic axial limit
(ρ → 1, |S| → 2) as its endpoint.

[Fig. 4: triangle vs cosine — ds3_iso.png]

## 5.5 Synthesis

The k_f family is a classical frontier: a one-parameter sweep of local models whose limit
point is the Bell bound (CHSH → 2), reached from inside the classical region. The quantum
target — an isotropic cosine with unit amplitude and CHSH = 2√2 — lies outside the family on
three independent axes at once: amplitude (bounded by ρ < 1), form (isotropy forces a triangle,
not a cosine), and isotropy itself (the cosine survives only along the privileged axis). No
single knob moves the ribbon toward it; that is the quantitative content of the phase-D bet's
outcome, and its interpretation is deferred to Section 7.
