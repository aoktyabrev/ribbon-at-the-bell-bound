# 2. Model

**Section owner:** Architect

## 2.1 Configuration space

The object is a framed discrete curve in R⁴: nodes x_i ∈ R⁴, i = 0..N−1,
with unit tangents t_i = (x_{i+1} − x_i)/|x_{i+1} − x_i| and the boundary
convention t_{N−1} := t_{N−2}. Each node carries a unit quaternion
u_i ∈ S³ ⊂ H, the lift of the node's normal frame relative to Bishop
(minimal-rotation) transport along the curve; the transport R_i ∈ SO(4)
maps t_i to t_{i+1} and acts as identity on the orthogonal complement of
their span. The full normal frame — not a single normal vector — is
what carries the topology: the normal space of a curve in R⁴ is
three-dimensional, π₁(SO(3)) = ℤ₂, and the ℤ₂ framing class lives in the
sign of the accumulated lift. A single normal (an S² of directions) or a
literal two-sided band (an SO(2) fiber) would carry no ℤ₂ invariant;
"ribbon" names the picture, "framed curve" names the object.

## 2.2 Energy

H = H_stretch + H_bend + H_frame + H_clamp, with
H_stretch = k_s Σ (|x_{i+1} − x_i| − ℓ)²,
H_bend    = k_b Σ (1 − t_i · t_{i+1})  (harmonic only near alignment),
H_frame   = k_f Σ d²(u_i, ū_{i+1}) where ū_{i+1} is u_{i+1} parallel-
transported back along the Bishop frame and d is the geodesic distance
on S³; the twist datum enters through arccos|⟨·,·⟩| and is therefore
blind to the lift sign by construction — a modeling choice, not a
consequence.
Two distinct clamp terms were used and must not be conflated. The frame
clamp, H_clamp = k_c · arccos²(|⟨u_end, U_target⟩|), pins the full
boundary frame and appears only in the D0 invariant-validation runs.
The axial clamp, H_clamp = −k_c (n_end · a)² with n_end the boundary
axis obtained from u_end under SU(2) → SO(3), pins an axis only — the
residual U(1) about it stays free — and is the clamp behind every
scientific number in this paper (D1 through the seed audit). The
axis–frame distinction is not a technicality: the U(1) theorem of
Section 5 lives in exactly the fiber the axial clamp leaves free. The frame
energy is blind to the lift sign by construction; the ℤ₂ invariant is
therefore carried by the kinematics — lift continuity between accepted
steps and singular-step rejection (2.3–2.4) — not by the energy.

## 2.3 Dynamics

Relaxation is a projected Euler–Maruyama scheme, stated as implemented:
per step, gradients g_x, g_u of H; noise ν ~ N(0,1)·√(2·lr·T) added in
the ambient space; for the frame variables the update du = −lr·g_u + ν
is projected onto the tangent space at u_i (du ← du − ⟨du,u⟩u) and
retracted by normalization. The step size lr doubles as the time
step; no metric correction, Itô–Stratonovich term, or retraction
Jacobian is applied; consequently we do not claim that the scheme
samples a Gibbs measure at temperature T, and no stationary measure is
derived — T is an algorithmic noise scale whose observed effect on the
reported amplitude is flat (Section 6). Steps are rejected for
the whole chain on any of three criteria: a temporal lift jump
(⟨u_new, u_old⟩ < 1 − δ_sing), a tangent reversal
(t_i · t_{i+1} < −1 + δ_tan), or a spatial lift wall — a sign change of
the link datum g.w across one step, the passage of a link through a
half-turn (δ_sing = δ_tan = 2·10⁻², band4d.py). The third criterion is
what protects the parity bookkeeping: it is the "leaky lift" detector,
and the rejection fraction falls with lr in the validated regime.
Lift continuity between accepted steps (the sign of u
chosen nearest the previous step) is what makes the parity bookkeeping
well-defined.

## 2.4 Preparation and topological invariant

The even sector is prepared as u ≡ const followed by relaxation; the odd
sector by a linear twist ramp distributing 2π along the chain, followed by
relaxation. The preparation is unbiased with respect to the outcome
branch: the two boundary frames are tilted about random axes with polar
angle arccos(1 − 2ξ), so that each end's axis n_end is uniform on S² — the
branch is never chosen in advance, and the basin decides
(prep_dynamics, measurement.py). Measurement settings a, b are inputs of
the protocol, fixed per run or drawn by the campaign design — never from
the preparation randomness; their independence from λ is a designed
property, used in 2.6(iii). The ℤ₂ parity of a
configuration is the sign of the lift accumulated along the chain with
continuous sign choice, compared against the boundary frames; it is
defined on trajectories free of rejected singular steps, is exactly
conserved on accepted trajectories, and is not spontaneously populated
thermally — charged configurations exist only by preparation.

## 2.5 Measurement and estimator

Outcomes are read locally at the boundaries: s = sign(n_0 · a),
t = sign(n_{N−1} · b), each a function of one end's frame and that end's
setting only. Configurations with |n · a| below a fixed threshold
(0.2, a convention) were classed DEGENERATE and reported as a separate
column in the campaigns; for CHSH estimation this class must not be
discarded — setting-dependent discard is precisely the detection
loophole documented in Section 7, and the corrected estimator
keeps all events with sign(0) → +1 by convention. The correlation
estimator is E(a,b) = (1/n_valid) Σ_{j: valid} s_j t_j with binomial
standard error √(max(1−E², ·)/n_valid), where valid excludes the
DEGENERATE class where it is reported separately; in the campaigns
entering CHSH and the seed audit, degen = 0 and n_valid = M. A
pre-registered cross-seed audit measured the seed-to-seed scatter against
that binomial error at r = 0.88 (11 repeats, N = 32 core cell) and
r = 1.11 (5 points, N = 96), so the quoted errors are honest as
reproducibility measures.

## 2.6 Locality status

Three statements at three strengths, kept separate deliberately.
(i) Readout locality holds by construction: s depends on (u_0, a) only,
t on (u_{N−1}, b) only. (ii) Dynamical locality is not claimed: both
clamps enter one global relaxation functional, so the final
configuration near one end may in general depend on the remote setting;
proving factorization P(s,t|a,b,λ) = P(s|a,λ)·P(t|b,λ) over the
relaxation dynamics would require its stationary measure, which is not
derived. (iii) Operationally, the isotropized protocol is a shared-λ
scheme: λ = (R, initial configuration and noise realization) with
R ~ Haar on SO(3) common to both ends, settings drawn independently of
λ, and local readout — and its measured behavior (the triangular
correlation, S ≤ 2 everywhere after audit) sits strictly inside the
local-hidden-variable region. The paper's Bell-side claims rest on (i)
and (iii); nothing rests on (ii).
