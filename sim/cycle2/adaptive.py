"""C2-J адаптивный интегратор (addendum-1 ветка Б). НОВЫЙ код-путь —
measurement.py/band4d.py НЕ трогаются (фаза D побитово сохранена).

Многообразный градиентный спуск при T=0 с per-replica Armijo line search:
энергия монотонна, шаг подстраивается под жёсткость (мягкая мода не
недорелаксирует, жёсткая не разносит). Ретракция u: normalize(u−α·g_tan).
Без искусственной отбраковки сингулярностей — сходимость сертифицируется
гейтом J0-v2 (max|grad|, инвариантность бассейнов).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ribbon_sim.frames import normalize
from measurement import e_meas


def build_adaptive_minimizer(params, alpha0=2e-3, alpha_max=0.1, c_armijo=1e-4,
                             n_backtrack=7, freeze_w=False):
    """T=0 Armijo-спуск e_meas. run(x0,u0,X0,XL,a,b,steps)→(x,u,grad_max)."""
    def e1(x, u, a, b):
        return e_meas(x, u, a, b, params)
    e_b = jax.vmap(e1, in_axes=(0, 0, None, None))              # (M,)
    gx_b = jax.vmap(jax.grad(e1, 0), in_axes=(0, 0, None, None))
    gu_b = jax.vmap(jax.grad(e1, 1), in_axes=(0, 0, None, None))

    def _fix_ends(x, X0, XL):
        x = x.at[:, 0, :].set(X0).at[:, -1, :].set(XL)
        if freeze_w:
            x = x.at[..., 0].set(0.0)
        return x

    def _grad_tan(x, u, a, b):
        gx = gx_b(x, u, a, b)
        gu = gu_b(x, u, a, b)
        gu = gu - jnp.sum(gu * u, axis=-1, keepdims=True) * u   # касательный к S³
        return gx, gu

    @partial(jax.jit, static_argnums=(6,))
    def run(x0, u0, X0, XL, a, b, steps):
        M = x0.shape[0]

        def step(carry, _):
            x, u, alpha = carry
            E = e_b(x, u, a, b)
            gx, gu = _grad_tan(x, u, a, b)
            gnorm2 = jnp.sum(gx**2, axis=(-1, -2)) + jnp.sum(gu**2, axis=(-1, -2))   # (M,)

            def trial(al):
                al_c = al[:, None, None]
                xn = _fix_ends(x - al_c * gx, X0, XL)
                un = normalize(u - al_c * gu)
                En = e_b(xn, un, a, b)
                ok = En <= E - c_armijo * al * gnorm2
                return ok

            # бэктрекинг от несомого alpha: alpha·0.5^k, берём НАИБОЛЬШИЙ с Armijo
            best = jnp.zeros(M)
            found = jnp.zeros(M, dtype=bool)
            for k in range(n_backtrack):
                al = alpha * (0.5 ** k)
                ok = trial(al)
                take = ok & (~found)
                best = jnp.where(take, al, best)
                found = found | ok
            # если ни один не прошёл — крошечный шаг (гарантирует не-возрастание при c малом)
            al_final = jnp.where(found, best, alpha * (0.5 ** (n_backtrack - 1)))
            ac = al_final[:, None, None]
            x_new = _fix_ends(x - ac * gx, X0, XL)
            u_new = normalize(u - ac * gu)
            # адаптация несомого alpha: рост если взят полный шаг, иначе — принятый
            alpha_new = jnp.clip(jnp.where(found & (best >= alpha), alpha * 1.5, al_final * 2.0),
                                 1e-9, alpha_max)
            return (x_new, u_new, alpha_new), None

        alpha0_arr = jnp.full((M,), alpha0)
        (xf, uf, _), _ = lax.scan(step, (x0, u0, alpha0_arr), None, length=steps)
        gxf, guf = _grad_tan(xf, uf, a, b)
        grad_max = jnp.sqrt(jnp.sum(gxf**2, axis=(-1, -2)) + jnp.sum(guf**2, axis=(-1, -2)))  # (M,)
        return xf, uf, grad_max

    return {"run": run, "alpha0": alpha0}
