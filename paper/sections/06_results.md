# 6. Quantitative boundary

**Section owner:** Code

## 6.1 The plateau

The one quantity the ribbon carries that does not dilute with length is the axial readout
amplitude of the odd sector. Fit against three models — a constant A_plateau, a power law A·N^−γ,
and a saturating A_plateau + c·N^−γ — the constant is substantially preferred among the three
tested models (ΔAICc = 6.36): A_plateau = 0.363 ± 0.012, with the power law's exponent fitted
negative (no decay) (D2-ext; commit 2784edf). We write A_plateau, and avoid A∞: the limit
N → ∞ is not measured; what is measured is no statistically resolved decay over 16 ≤ N ≤ 96. The plateau holds at a second stiffness: cross-scanning N ∈ {16, 32, 64, 96}
at k_f×1 and k_f×4, the amplitude change from N = 16 to N = 96 is +0.010 ± 0.038 and
+0.002 ± 0.023 respectively — flat within error at both couplings (DS2; commit f928dd4). The evidence is
bounded: N ∈ [16, 96], k_f ∈ {×1, ×4}. Within that window the plateau is kinetic, not
thermodynamic — there is no statistically resolved temperature dependence over
T ∈ {0.025, 0.05, 0.10} (A(T) = 0.368 / 0.397 / 0.353, within σ ≈ 0.027) (S1-runs;
commit a9cef7b) — so it is a property of the basin structure, not of thermal fluctuation. The quoted errors are binomial,
and a dedicated seed audit confirms they are honest as a measure of reproducibility: eleven
independent repeats of the N = 32 cell scatter with s_seed = 0.024 against a binomial
σ = 0.027, a ratio r = 0.88 (χ² = 7.7 / 10, p = 0.66), and the ratio does not grow with chain
length (r = 1.11 at N = 96) (seed audit; commit a26f76b).

![Fig. 1a: D2-ext scaling A_N vs N](../../sim/phase_D/fig/d2ext_scaling.png)

![Fig. 1b: DS2 cross-scan at two stiffnesses](../../sim/phase_D/fig/ds2_cross.png)

*Fig. 1. The plateau. (a) D2-ext: A_N = |E(0)| against chain length N ∈ {16, 32, 48, 64, 96}
at M = 1200 replicas, with the three fitted models — constant M0 (A_plateau = 0.363, dotted), power
law M1 (γ = −0.03, i.e. fitted growth rather than decay, dashed), and saturating M2 (solid).
M0 is substantially preferred by ΔAICc = 6.36 (commit 2784edf); M2 is degenerate on this data
and shown for completeness. (b) DS2 cross-scan: A(N) at k_f×1 and k_f×4, flat within error at
both stiffnesses (commit f928dd4).*

## 6.2 Origin of the amplitude

The amplitude is stiffness-controlled. Sweeping the twist coupling gives
A(k_f) = 0.27 / 0.42 / 0.56 / 0.84 for k_f × {0.5, 1, 2, 4} — a strong
monotone dependence that refutes any fixed geometric-measure origin
(S1-runs R1; commit a9cef7b). The plateau value A_plateau = 0.363 ± 0.012 belongs
to the D2-ext campaign (N ∈ [16, 96] at its baseline setup); the stiffness
sweep, run at N = 32, gives 0.418 ± 0.026 at nominal k_f×1 — consistent in
scale, not identical — the gap reflects seed-to-seed sampling scatter,
quantified in the seed audit (s_seed = 0.024). This only sharpens the point: the amplitude is the
reading of a dial, not a constant of the model; the weakness of the signal
is a property of soft coupling, not a limit of the mechanism. At θ = 0 the
amplitude is, by identity, A = 2·P_aligned − 1 (P_aligned = 0.681 for the
D2-ext plateau), so explaining the amplitude means explaining the
aligned-basin weight. Two measured facts locate that
weight in kinetics rather than thermodynamics. First, it is flat in
temperature (Section 6.1), while any Boltzmann reading of the landscape
would move with T: the census finds the four branch minima spread over
0.22 in energy (branch means −3.78 for both branches, Δ ≈ 0.006), and no
assignment of Boltzmann weights to these energies reproduces a
68/32 split with no resolved temperature dependence (D1 census; D_synthesis_S1 §2).
Second, it is flat in chain length from N = 16 to N = 96 at both measured
stiffnesses (Section 6.1), while an equilibrium chain correlation with a
finite correlation length would decay. What remains is relaxation itself:
the split is set by the geometry of the attraction domains under the
clamped dynamics — a domain volume controlled by coupling stiffness. This
is the reading fixed in the canonical record after the cross-scan (commit
311d3ae): the amplitude is a stiffness-controlled kinetic correlation,
athermal and sector-blind (Section 5.3).

![Fig. 2: A_plateau(k_f) — stiffness memory](../../sim/phase_D/fig/s1runs_kf.png)

*Fig. 2. A_plateau(k_f) — stiffness memory. Four points with 1σ error bars from the frozen
S1-runs R1 raw data (N = 32, M = 1200, T = 0.05; commit a9cef7b). Dotted line: the
D2-ext plateau A_plateau = 0.363 of Section 6.1, measured over N ∈ [16, 96].*

## 6.3 Anisotropy map

The cosine angular law the correlation appears to follow is anisotropic. Tilting the clamp axis
a by α from the privileged axis ê and reading the antiparallel amplitude A(α) = |E_anti|, the
signal decays smoothly to zero at α = π/2: from 0.387 to 0.005 at k_f×1 and from 0.865 to 0.008
at k_f×4 (DS3; commit 0fb5452). The best-fit law is cos α at k_f×1 and steepens to ~cos²α (an
exponential fitting equally well) at k_f×4. The axis ê — simultaneously the twist axis and the
readout axis — is strongly privileged; the cosine dependence exists only along it, which is
exactly why the isotropic CHSH [CHSH1969] estimator of Section 7 was invalid.

![Fig. 3: DS3 anisotropy map](../../sim/phase_D/fig/ds3_aniso.png)

*Fig. 3. Anisotropy map. A(α) = |E_anti| as the clamp axis a is tilted by α from the
privileged axis ê, at k_f×1 (best fit cos α) and k_f×4 (best fit exp ≈ cos²α); dashed curves
are cos²α. The signal decays to zero at α = π/2 at both stiffnesses: the cosine law exists
only along ê (commit 0fb5452).*

## 6.4 The trilemma and the triangle

Averaged honestly over the shared randomness λ = (R, n_A, n_B) — where R ~ Haar on SO(3) is
the ribbon's orientation, common to both ends through the geometry, while the settings n_A and
n_B are drawn independently — the correlation is not a cosine. Fitting the
isotropized E(θ) against both forms, the triangular local-realistic function E = −ρ(1 − 2θ/π)
beats the cosine by a wide margin in χ²: 8 vs 39 at k_f×1 and 1 vs 218 at k_f×4 — the data lie
on the straight line, not the curve (DS3; commit 0fb5452). This is the shared-λ local model's
signature [Bell1964], and its CHSH value is exactly S = 2ρ, with ρ the source-alignment amplitude:
ρ = 0.374 / 0.810 gives |S| = 0.75 / 1.62 at k_f × {1, 4}. Three properties therefore trade
off and cannot be held together: a cosine form exists only anisotropically (6.3); honest
isotropy forces the triangle (this subsection); and the amplitude ρ is bought with stiffness
(6.2). The measured family moves toward the Bell bound as stiffness grows: |S| = 1.62 at
k_f×4 sits a measured distance 2 − 1.62 = 0.38 short of it. Its deterministic axial limit is
expected — not measured — to reach ρ → 1, |S| → 2.

![Fig. 4: DS3 isotropized correlation — triangle vs cosine](../../sim/phase_D/fig/ds3_iso.png)

*Fig. 4. Triangle versus cosine. The honestly isotropized E(θ) at k_f×1 and k_f×4: the data
lie on the straight triangular law −ρ(1 − 2θ/π) (solid, ρ = 0.374 / 0.810, CHSH = 2ρ =
0.75 / 1.62), not on the cosine (dotted). Horizontal reference lines mark the literature
values 1/3, 2/π, 1/K_G, which were formulated for a cosine LHV and are not direct bounds on
a triangular one (DS3 §4; commit 0fb5452).*

## 6.5 Synthesis

The k_f family is a classical frontier: a one-parameter sweep of local models measured from
inside the classical region, moving toward the Bell bound as stiffness grows and expected —
not measured — to reach CHSH → 2 in its deterministic axial limit. The quantum
target — an isotropic cosine with unit amplitude and CHSH = 2√2 — lies outside the family on
three independent axes at once: amplitude (bounded by ρ < 1), form (isotropy forces a triangle,
not a cosine), and isotropy itself (the cosine survives only along the privileged axis). No
single knob moves the ribbon toward it; that is the quantitative content of the phase-D bet's
outcome, and its interpretation is deferred to Section 8.
