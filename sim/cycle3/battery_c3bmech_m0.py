"""C3-B-mech / M0 калибровка источника (prereg d691435). GPU/JAX.
Prep-энергия = упругость (band4d) − h·Σ_i(n_i·ẑ) (поляризующее поле, prep-only),
БЕЗ измерит. зажимов. Релаксация батча (отжиг И квенч) → распределение n_mid;
фит χ(h) = 3·⟨n_mid,z⟩ (⟨c⟩=χ/3 для меры 1+χc). KILL: χ<3σ ∀h. --smoke.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

_SIM = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_SIM, "phase_D"))
sys.path.insert(0, os.path.join(_SIM, "src"))
import band4d as B
import measurement as M
from ribbon_sim.frames import normalize

SMOKE = "--smoke" in sys.argv
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")
PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, ell=1.0)     # без k_c (нет зажимов в prep)
N = 24 if SMOKE else 32
BATCH = 256 if SMOKE else 2000
H_GRID = [0.0, 0.02, 0.05, 0.1, 0.2]     # линейный режим (форма 1+χn_z ⇒ χ≤1)
LR, T_HOT, T_PREP = 5e-3, 2.0, 0.5      # addendum1 A-T1: T_prep=0.5
Z = jnp.array([0.0, 0.0, 1.0])
axis_all = jax.vmap(M.end_axis)  # (N,4)->(N,3)


def e_prep(x, u, h):
    """Упругость D0 (band4d) − h·Σ_i (n_i·ẑ). Чистая от сырых (x,u)."""
    t = B.node_tangents(x)
    n_z = axis_all(u)[:, 2]                              # (N,) проекции осей на ẑ
    return (B.e_stretch(x, PARAMS["k_s"], PARAMS["ell"])
            + B.e_bend(t, PARAMS["k_b"])
            + B.e_twist(u, t, PARAMS["k_f"])
            - h * jnp.sum(n_z))


def build_relaxer(h):
    gx = jax.vmap(jax.grad(lambda x, u: e_prep(x, u, h), 0))
    gu = jax.vmap(jax.grad(lambda x, u: e_prep(x, u, h), 1))

    @jax.jit
    def run(key, x0, u0, X0, XL, T_sched):
        def step(carry, inp):
            x, u, rej = carry
            T, k = inp
            kx, ku = jr.split(k)
            g_x = gx(x, u); g_u = gu(x, u)
            nx = jr.normal(kx, x.shape) * jnp.sqrt(2.0 * LR * T)
            nu = jr.normal(ku, u.shape) * jnp.sqrt(2.0 * LR * T)
            x_new = (x - LR * g_x + nx).at[:, 0].set(X0).at[:, -1].set(XL)
            du = -LR * g_u + nu
            du = du - jnp.sum(du * u, axis=-1, keepdims=True) * u   # касательный к S³
            u_new = normalize(u + du)
            flip = jnp.sign(jnp.sum(u_new * u, axis=-1, keepdims=True) + 1e-30)
            u_new = u_new * flip
            # rejection-мера: реверс касательной (гейт стабильности A3)
            t_old = jax.vmap(B.node_tangents)(x); t_new = jax.vmap(B.node_tangents)(x_new)
            tdot = jnp.sum(t_new[:, :-1] * t_new[:, 1:], axis=-1)
            bad = jnp.any(tdot < -1 + B.DELTA_TAN, axis=-1)
            x_out = jnp.where(bad[:, None, None], x, x_new)
            u_out = jnp.where(bad[:, None, None], u, u_new)
            return (x_out, u_out, rej + bad.astype(rej.dtype)), None
        keys = jr.split(key, T_sched.shape[0])
        (x, u, rej), _ = jax.lax.scan(step, (x0, u0, jnp.zeros(x0.shape[0])),
                                      (T_sched, keys))
        return x, u, rej
    return run


def sched_plateau():
    """отжиг T_hot→T_prep + ПЛАТО при T_prep (addendum1 A-T1 шаг 1)."""
    ramp = (400 if SMOKE else 3000)
    plat = (300 if SMOKE else 2000)
    return jnp.concatenate([jnp.linspace(T_HOT, T_PREP, ramp), jnp.full(plat, T_PREP)]), plat


def sched_freeze():
    """квенч T_prep→0 БЕЗ поля (addendum1 A-T1 шаг 2)."""
    return jnp.linspace(T_PREP, 0.0, (300 if SMOKE else 2000))


def fit_form(c):
    """Фит плотности ∝ 1+χ·c на гистограмме cosθ=n_z; χ²/dof. Возврат (χ_fit, chi2dof)."""
    bins = 20; w = 2.0 / bins
    hist, edges = np.histogram(c, bins=bins, range=(-1, 1), density=True)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    # плотность на S² в cosθ равномерна ⇒ p(c)=(1+χc)/2; взвеш. МНК: y=2p−1=χc.
    y = 2 * hist - 1.0
    var_p = np.maximum(hist, 1e-3) / (len(c) * w)      # var плотностного бина
    var_y = 4 * var_p
    chi = float(np.sum(ctr * y / var_y) / (np.sum(ctr * ctr / var_y) + 1e-12))
    chi2 = float(np.sum((y - chi * ctr) ** 2 / var_y) / max(bins - 1, 1))
    return chi, chi2


def n_mid_z(u):
    """n_mid,z на ленту: средняя проекция осей центральной трети на ẑ, батч (B,N,4)->(B,)."""
    lo, hi = N // 3, 2 * N // 3
    axes = jax.vmap(axis_all)(u)               # (B,N,3)
    mid = axes[:, lo:hi, :].mean(1)            # (B,3)
    mid = mid / (jnp.linalg.norm(mid, axis=-1, keepdims=True) + 1e-12)
    return mid[:, 2]


def run_M0():
    prep = M.prep_dynamics(N, False, BATCH, jr.PRNGKey(20260721))
    sp, plat = sched_plateau(); sf = sched_freeze()
    rows = {}
    print(f"\n  Протокол: отжиг T_hot→T_prep={T_PREP} + плато({plat}) [поле h] → квенч→0 [поле off]")
    print(f"  {'h':>6} {'χ_plat=3⟨c⟩':>12} {'σ':>8} {'фит χ':>8} {'χ²/dof':>8} "
          f"{'χ_froz':>8} {'froz/plat':>9} {'rej/step':>10} {'эрго':>6}")
    for h in H_GRID:
        run_field = build_relaxer(h)
        x, u, rej = run_field(jr.PRNGKey(7), prep["x0"], prep["u0"], prep["X0"], prep["XL"], sp)
        c = np.asarray(n_mid_z(u))
        chi_p = 3.0 * float(c.mean()); sig = 3.0 * float(c.std()) / np.sqrt(len(c))
        chi_fit, chi2dof = fit_form(c)
        rejps = float(np.asarray(rej).sum()) / (BATCH * sp.shape[0])
        # эргодичность: дрейф ⟨c⟩ за последние 20% плато — оценка через доп. короткий прогон
        x2, u2, _ = build_relaxer(h)(jr.PRNGKey(8), x, u, prep["X0"], prep["XL"],
                                     jnp.full(max(1, plat // 5), T_PREP))
        drift = abs(3.0 * float(np.asarray(n_mid_z(u2)).mean()) - chi_p)
        ergo = drift < sig
        # заморозка: поле off, квенч T_prep→0 от плато-конфига
        xf, uf, _ = build_relaxer(0.0)(jr.PRNGKey(9), x, u, prep["X0"], prep["XL"], sf)
        chi_f = 3.0 * float(np.asarray(n_mid_z(uf)).mean())
        ratio = chi_f / chi_p if abs(chi_p) > 3 * sig else float("nan")
        rows[f"{h}"] = dict(chi_plateau=chi_p, sigma=sig, chi_fit=chi_fit, chi2dof=chi2dof,
                            chi_frozen=chi_f, frozen_over_plateau=ratio, rej_per_step=rejps,
                            ergodic=bool(ergo), stable=bool(rejps < 1e-3), form_ok=bool(chi2dof < 2))
        print(f"  {h:>6} {chi_p:>12.4f} {sig:>8.4f} {chi_fit:>8.4f} {chi2dof:>8.3f} "
              f"{chi_f:>8.4f} {ratio:>9.3f} {rejps:>10.2e} {str(ergo):>6}")
    return rows


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N}, BATCH={BATCH}, T_prep={T_PREP}, GPU={jax.devices()}")
    rows = run_M0()
    # h_max (A-T3): наибольший h с rej/step<1e-3 И form_ok на плато
    stable_h = [float(h) for h, v in rows.items() if v["stable"] and v["form_ok"]]
    # технический KILL (A-T4): форма 1+χn_z не фитится где-либо в валидном диапазоне
    form_fail = [h for h, v in rows.items() if float(h) > 0 and not v["form_ok"] and v["stable"]]
    tag = "_smoke" if SMOKE else ""
    json.dump(dict(rows=rows, params=PARAMS, N=N, batch=BATCH, T_prep=T_PREP,
                   h_grid=H_GRID, h_max=(max(stable_h) if stable_h else None),
                   form_fail_h=form_fail),
              open(os.path.join(RES, f"C3Bmech_M0{tag}.json"), "w"), indent=2)
    print(f"\n  h_max (rej<1e-3 ∧ форма χ²/dof<2, плато): {max(stable_h) if stable_h else 'нет'}")
    print(f"  калибровка χ_plateau(h) ≈ h/T_prep (Больцман); "
          f"технический KILL (форма не фитится): {form_fail if form_fail else 'нет'}")
    print(f"  → {RES}/C3Bmech_M0{tag}.json")


if __name__ == "__main__":
    main()
