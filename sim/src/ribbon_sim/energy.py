"""Энергия ленты (SPEC §2.2).

Фаза 1 — изотропная упругость: сумма квадратов геодезических расстояний
между соседними кадрами. Зажимы концов — квадратичные (±-симметричные).
"""

import jax.numpy as jnp

from .frames import axis, geodesic


def e_elastic(q, k_e, spinor=False):
    """E_elastic = k_e * Σ_{i=0}^{N-2} d(q[i], q[i+1])²   (SPEC §2.2).

    q формы (N, 4). Изотропная фаза 1: все соседние связи равноправны.
    """
    d = geodesic(q[:-1], q[1:], spinor=spinor)  # (N-1,)
    return k_e * jnp.sum(d * d)


def e_clamp(q_end, axis_vec, k_c):
    """E_clamp = −k_c * (n·axis)²   (SPEC §2.2).

    Квадратичный зажим: два минимума при n = ±axis, знак не предпочтён.
    """
    n = axis(q_end)  # (3,)
    proj = jnp.dot(n, axis_vec)
    return -k_c * proj * proj


def e_total(q, a, b, k_e, k_c, spinor=False):
    """Полная энергия одной ленты (SPEC §2.2):

        E_total = E_elastic + E_clamp_A + E_clamp_B

    q формы (N, 4); a, b — единичные 3-векторы осей зажимов; скаляр на выходе.
    Функция чистая от СЫРОГО q (без внутренней нормировки) — так конечные
    разности и jax.grad видят одну и ту же функцию (тест SPEC §7).
    """
    return (
        e_elastic(q, k_e, spinor=spinor)
        + e_clamp(q[0], a, k_c)
        + e_clamp(q[-1], b, k_c)
    )
