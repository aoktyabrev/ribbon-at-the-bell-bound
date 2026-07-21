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
H_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
LR, T_HOT = 5e-3, 2.0
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


def sched(regime):
    """отжиг: T_hot→0 линейно + доводка; квенч: сразу T=0 (быстрый спуск)."""
    steps = (300 if SMOKE else 1500)
    if regime == "anneal":
        slow = 16 if not SMOKE else 4
        ramp = steps * slow // 4
        return jnp.concatenate([jnp.linspace(T_HOT, 0.0, ramp), jnp.zeros(ramp // 4)])
    return jnp.zeros(steps)   # квенч: прямой T=0 спуск


def n_mid_z(u):
    """n_mid,z на ленту: средняя проекция осей центральной трети на ẑ, батч (B,N,4)->(B,)."""
    lo, hi = N // 3, 2 * N // 3
    axes = jax.vmap(axis_all)(u)               # (B,N,3)
    mid = axes[:, lo:hi, :].mean(1)            # (B,3)
    mid = mid / (jnp.linalg.norm(mid, axis=-1, keepdims=True) + 1e-12)
    return mid[:, 2]


def run_regime(regime):
    prep = M.prep_dynamics(N, False, BATCH, jr.PRNGKey(20260721))
    T_sched = sched(regime)
    rows = {}
    print(f"\n  [{regime}] steps={T_sched.shape[0]}  h-сетка {H_GRID}")
    print(f"  {'h':>6} {'χ=3⟨c⟩':>10} {'σ_χ':>9} {'⟨c⟩':>9} {'rej/step':>10} {'>3σ':>5}")
    for h in H_GRID:
        run = build_relaxer(h)
        x, u, rej = run(jr.PRNGKey(7), prep["x0"], prep["u0"], prep["X0"], prep["XL"], T_sched)
        c = np.asarray(n_mid_z(u))
        chi = 3.0 * float(c.mean()); sig = 3.0 * float(c.std()) / np.sqrt(len(c))
        rejps = float(np.asarray(rej).sum()) / (BATCH * T_sched.shape[0])
        ok = abs(chi) > 3 * sig
        rows[f"{h}"] = dict(chi=chi, sigma=sig, mean_c=float(c.mean()),
                            rej_per_step=rejps, above_3sig=bool(ok), stable=bool(rejps < 1e-3))
        print(f"  {h:>6} {chi:>10.4f} {sig:>9.4f} {c.mean():>9.4f} {rejps:>10.2e} {str(ok):>5}")
    return rows


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N}, BATCH={BATCH}, GPU={jax.devices()}")
    anneal = run_regime("anneal"); quench = run_regime("quench")
    # KILL: χ<3σ ∀h (в отжиге — первичный режим)
    wall = all(not v["above_3sig"] for v in anneal.values())
    # h_max по гейту стабильности (A3): наибольший h с rej/step<1e-3
    stable_h = [float(h) for h, v in anneal.items() if v["stable"]]
    tag = "_smoke" if SMOKE else ""
    json.dump(dict(anneal=anneal, quench=quench, params=PARAMS, N=N, batch=BATCH,
                   h_grid=H_GRID, wall_M0=wall, stable_h=stable_h),
              open(os.path.join(RES, f"C3Bmech_M0{tag}.json"), "w"), indent=2)
    print(f"\n  h_max (rej/step<1e-3, отжиг): {max(stable_h) if stable_h else 'нет'}")
    print(f"  KILL M0 (χ<3σ ∀h, отжиг): {'СТЕНА — источник НЕ поляризуется' if wall else 'нет — источник поляризуется'}")
    print(f"  → {RES}/C3Bmech_M0{tag}.json")


if __name__ == "__main__":
    main()
