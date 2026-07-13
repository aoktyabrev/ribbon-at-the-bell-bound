"""Unit-тесты ℤ₂-инварианта parity (D0 §2): T-inv-1/2/3."""

import jax.numpy as jnp
import jax.random as jr

import band4d as B
import invariant as I
from ribbon_sim.frames import normalize

_ID = jnp.array([1.0, 0.0, 0.0, 0.0])


def test_inv1_straight_identity_plus():
    """T-inv-1: прямой стержень, тождественный репер → +1."""
    N = 24
    x = I.straight_rod(N)
    u = jnp.tile(_ID, (N, 1))
    assert float(I.parity(x, u, _ID, _ID)) == 1.0


def test_inv2_2pi_minus_4pi_plus():
    """T-inv-2: 2π → −1; 4π → +1 (сверка с тождественным аппаратным репером U_B=id).

    Аппаратный репер U_B — тождественный лифт (1,0,0,0): 2π-поворот приводит
    непрерывный лифт эндпоинта в антипод (−1,0,0,0) ⇒ рассогласование ⇒ −1; 4π возвращает.
    """
    N = 32
    x = I.straight_rod(N)
    for axis in [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (0.3, -0.7, 0.5)]:
        u2 = I.frame_rotation(N, 2 * jnp.pi, axis3=axis)
        u4 = I.frame_rotation(N, 4 * jnp.pi, axis3=axis)
        assert float(I.parity(x, u2, _ID, _ID)) == -1.0
        assert float(I.parity(x, u4, _ID, _ID)) == 1.0


def test_inv3_robust_to_smooth_deformation():
    """T-inv-3: 1000 малых гладких деформаций без зажимов — parity не меняется ни разу;
    сингулярные шаги отбракованы и посчитаны (счётчик > 0).

    Деформируем ИНТЕРЬЕР (концы = фиксированные граничные данные краевой задачи),
    каждая деформация независима и мала. Проверяем все три сектора (0, 2π, 4π)."""
    N = 24
    x0 = I.straight_rod(N)

    def smooth_pert(key, shape, scale, kmodes=3):
        # низкочастотная (гладкая) деформация вдоль цепочки
        Nn = shape[0]
        ks = jr.split(key, 3 * kmodes)
        s = jnp.arange(Nn) / (Nn - 1)
        out = jnp.zeros(shape)
        for m in range(1, kmodes + 1):
            amp = jr.normal(ks[3 * m - 3], (1,) + shape[1:])
            phase = jr.uniform(ks[3 * m - 2]) * 2 * jnp.pi
            out = out + jnp.sin(m * jnp.pi * s + phase)[:, None] * amp
        return out * scale

    def hold(a, a0):
        a = a.at[0].set(a0[0])
        return a.at[-1].set(a0[-1])

    total_rejected = 0
    for tot in [0.0, 2 * jnp.pi, 4 * jnp.pi]:
        u0 = I.frame_rotation(N, tot)
        U_A, U_B = u0[0], u0[-1]
        p0 = float(I.parity(x0, u0, U_A, U_B))
        key = jr.PRNGKey(11)
        rejected = 0
        for _ in range(1000):
            key, ka, kb = jr.split(key, 3)
            dx = smooth_pert(ka, (N, 4), 0.04).at[:, 0].set(0.0)  # держим в срезе
            du = smooth_pert(kb, (N, 4), 0.04)
            x = hold(x0 + dx, x0)
            radial = jnp.sum(du * u0, axis=-1, keepdims=True)
            u = hold(normalize(u0 + (du - radial * u0)), u0)
            # детекция сингулярностей (та же логика, что в band4d.build_stepper)
            overlap = jnp.sum(u * u0, axis=-1)
            t = B.node_tangents(x)
            tdot = jnp.sum(t[:-1] * t[1:], axis=-1)
            if bool(jnp.any(overlap < 1 - B.DELTA_SING) or jnp.any(tdot < -1 + B.DELTA_TAN)):
                rejected += 1
                continue
            assert float(I.parity(x, u, U_A, U_B)) == p0   # ни разу не меняется
        total_rejected += rejected
    assert total_rejected > 0   # сингулярные шаги реально встречались и посчитаны
