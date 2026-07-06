"""Батч-релаксация: ланжевен с геометрическим отжигом (SPEC §2.3).

Чистый JAX: jit(vmap(grad)) по батчу лент + lax.scan по шагам. Никаких
питоновских циклов по шагам/батчу. spinor/elastic — статические (замыкание),
чтобы не тащить их через vmap как трассируемые аргументы.
"""

import jax
import jax.numpy as jnp
from jax import lax

from .energy import e_total
from .frames import axis, normalize


def build_relaxer(cfg):
    """Собирает jit-функции релаксации и зонда сходимости по конфигу.

    cfg — dict: lr, T0, decay, steps, k_e, k_c, spinor(опц), elastic(опц).
    steps статичен (длина lax.scan). Возврат dict:
      - run(key, q0, a, b) → (q_final (B,N,4), e_trace (steps,) — средняя по батчу);
      - probe(q, a, b) → |ΔE| на ленту за один шаг T=0 (B,) — критерий сходимости.
    a, b трассируемы ⇒ одна компиляция на весь свип θ.
    """
    lr = float(cfg["lr"])
    T0 = float(cfg["T0"])
    decay = float(cfg["decay"])
    steps = int(cfg["steps"])
    k_e = float(cfg["k_e"])
    k_c = float(cfg["k_c"])
    spinor = bool(cfg.get("spinor", False))
    elastic = str(cfg.get("elastic", "geodesic"))

    # Замыкаем spinor/elastic в чистую функцию энергии одной ленты.
    def e_fn(q, a, b):
        return e_total(q, a, b, k_e, k_c, spinor=spinor, elastic=elastic)

    grad_batch = jax.vmap(jax.grad(e_fn), in_axes=(0, None, None))
    energy_batch = jax.vmap(e_fn, in_axes=(0, None, None))

    @jax.jit
    def run(key, q0, a, b):
        def step(carry, i):
            q, k = carry
            k, sub = jax.random.split(k)
            # SPEC §2.3: геометрический отжиг T = T0 * decay^step.
            T = T0 * decay ** i
            g = grad_batch(q, a, b)
            # q += −lr*grad + sqrt(2*lr*T)*noise, затем проекция на S³.
            noise = jax.random.normal(sub, q.shape) * jnp.sqrt(2.0 * lr * T)
            q = normalize(q - lr * g + noise)
            e = jnp.mean(energy_batch(q, a, b))
            return (q, k), e

        (q_final, _), e_trace = lax.scan(step, (q0, key), jnp.arange(steps))
        return q_final, e_trace

    @jax.jit
    def probe(q, a, b):
        # Один детерминированный (T=0) шаг: |ΔE| на ленту как мера остаточного дрейфа.
        e0 = energy_batch(q, a, b)
        g = grad_batch(q, a, b)
        q1 = normalize(q - lr * g)
        e1 = energy_batch(q1, a, b)
        return jnp.abs(e1 - e0)

    return {"run": run, "probe": probe, "lr": lr, "steps": steps}


def classify(q_final, a, b):
    """Финальная классификация ветвей (SPEC §2.3):

        s = sign(n_A · a),  t = sign(n_B · b).

    q_final формы (B, N, 4) → (s (B,), t (B,)) со значениями ±1.
    sign(0) доопределяем как +1 (нулевая проекция — мера нуль).
    """
    n_a = axis(q_final[:, 0])
    n_b = axis(q_final[:, -1])
    s = jnp.where(jnp.sum(n_a * a, axis=-1) >= 0, 1, -1)
    t = jnp.where(jnp.sum(n_b * b, axis=-1) >= 0, 1, -1)
    return s, t


def branch_counts(s, t):
    """Счётчики четырёх ветвей (SPEC §4.1): (n_pp, n_pm, n_mp, n_mm)."""
    n_pp = jnp.sum((s > 0) & (t > 0))
    n_pm = jnp.sum((s > 0) & (t < 0))
    n_mp = jnp.sum((s < 0) & (t > 0))
    n_mm = jnp.sum((s < 0) & (t < 0))
    return jnp.array([n_pp, n_pm, n_mp, n_mm])
