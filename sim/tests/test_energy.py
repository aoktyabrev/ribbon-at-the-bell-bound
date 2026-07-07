"""Тесты энергии (SPEC §7): минимумы зажима и градиент против конечных разностей."""

import jax

jax.config.update("jax_enable_x64", True)  # x64 ради точности конечных разностей

import jax.numpy as jnp
import numpy as np
import pytest

from ribbon_sim.energy import e_clamp, e_cosserat, e_cosserat_chordal, e_elastic, e_total
from ribbon_sim.frames import axis, haar_quaternions, normalize, quat_mul

_J = jnp.array([0.0, 1.0, 0.0, 0.0])  # 180° вокруг x: флип оси n→−n (q→q⊗j)


def _grad_elastic(mode, q, kb=2.0, kt=0.5, ke=63.0):
    if "cosserat" in mode:
        f = lambda z: e_elastic(z, 0.0, elastic=mode, k_b=kb, k_t=kt)  # noqa: E731
    else:
        f = lambda z: e_elastic(z, ke, elastic=mode)  # noqa: E731
    return jax.grad(f)(q)


def _unit(v):
    return v / jnp.linalg.norm(v)


def test_clamp_minima_at_plus_minus_a():
    a = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    k_c = 1.3
    # n = +a: тождество; n = −a: поворот на π вокруг x
    q_plus = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    q_minus = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float64)
    assert np.allclose(axis(q_plus), a, atol=1e-9)
    assert np.allclose(axis(q_minus), -a, atol=1e-9)
    # оба — минимумы со значением −k_c
    assert np.allclose(float(e_clamp(q_plus, a, k_c)), -k_c, atol=1e-9)
    assert np.allclose(float(e_clamp(q_minus, a, k_c)), -k_c, atol=1e-9)


def test_clamp_is_global_minimum():
    a = _unit(jnp.array([0.2, -0.5, 0.8], dtype=jnp.float64))
    k_c = 1.0
    qs = haar_quaternions(jax.random.PRNGKey(11), (5000,))
    e = jax.vmap(lambda q: e_clamp(q, a, k_c))(qs.astype(jnp.float64))
    # (n·a)² ≤ 1 ⇒ E ≥ −k_c для любого q
    assert np.all(np.asarray(e) >= -k_c - 1e-9)
    # минимум достигается (есть q с E ≈ −k_c)
    assert float(jnp.min(e)) < -k_c + 1e-2


def test_clamp_sign_symmetry():
    # квадратичный зажим ±-симметричен: E(q) = E(антипод оси)
    a = _unit(jnp.array([0.3, 0.4, 0.5], dtype=jnp.float64))
    qs = haar_quaternions(jax.random.PRNGKey(12), (100,)).astype(jnp.float64)
    e_q = jax.vmap(lambda q: e_clamp(q, a, 1.0))(qs)
    e_flip = jax.vmap(lambda q: e_clamp(q, -a, 1.0))(qs)
    assert np.allclose(np.asarray(e_q), np.asarray(e_flip), atol=1e-10)


def test_grad_matches_finite_differences():
    """jax.grad(E_total) против центральных конечных разностей, rtol 1e-4 (SPEC §7)."""
    N = 6
    key = jax.random.PRNGKey(20)
    q = haar_quaternions(key, (N,)).astype(jnp.float64)
    a = _unit(jnp.array([0.1, 0.2, 0.97], dtype=jnp.float64))
    b = _unit(jnp.array([0.8, -0.1, 0.4], dtype=jnp.float64))
    k_e, k_c = 0.7, 1.1

    g = jax.grad(e_total)(q, a, b, k_e, k_c, False)

    h = 1e-6
    fd = np.zeros_like(np.asarray(q))
    qn = np.asarray(q)
    for i in range(N):
        for j in range(4):
            qp = qn.copy(); qp[i, j] += h
            qm = qn.copy(); qm[i, j] -= h
            ep = float(e_total(jnp.array(qp), a, b, k_e, k_c, False))
            em = float(e_total(jnp.array(qm), a, b, k_e, k_c, False))
            fd[i, j] = (ep - em) / (2 * h)

    assert np.allclose(np.asarray(g), fd, rtol=1e-4, atol=1e-6)


def test_elastic_zero_for_aligned_chain():
    # все кадры одинаковы ⇒ E_elastic ≈ 0
    q = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64), (10, 1))
    assert float(e_elastic(q, k_e=1.0)) < 1e-4


def test_elastic_positive_and_scales_with_ke():
    q = haar_quaternions(jax.random.PRNGKey(21), (10,)).astype(jnp.float64)
    e1 = float(e_elastic(q, k_e=1.0))
    e2 = float(e_elastic(q, k_e=2.0))
    assert e1 > 0
    assert np.allclose(e2, 2 * e1, rtol=1e-9)


def test_chordal_elastic_properties():
    # chordal = 1 − <p,q>²: ноль на совпадении, ±-симметрия, положительность
    q_aligned = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64), (8, 1))
    assert abs(float(e_elastic(q_aligned, 1.0, elastic="chordal"))) < 1e-12
    q = haar_quaternions(jax.random.PRNGKey(30), (8,)).astype(jnp.float64)
    assert float(e_elastic(q, 1.0, elastic="chordal")) > 0
    # ±-симметрия: инверсия знака любого кадра не меняет chordal-энергию
    q_flip = q.at[3].multiply(-1.0)
    assert np.allclose(float(e_elastic(q, 1.0, elastic="chordal")),
                       float(e_elastic(q_flip, 1.0, elastic="chordal")), atol=1e-10)


def test_geodesic_grad_bitwise_equivariant():
    """geodesic: grad(q⊗j) == transform(grad(q)) ПОБИТОВО (atol=0). Осевая ±-симметрия."""
    q = haar_quaternions(jax.random.PRNGKey(60), (10,))
    g = _grad_elastic("geodesic", q)
    gj = _grad_elastic("geodesic", quat_mul(q, _J))
    assert np.array_equal(np.asarray(gj), np.asarray(quat_mul(g, _J)))


def test_cosserat_chordal_grad_equivariant_tight():
    """cosserat_chordal: НЕ побитово (quat_mul суммирует 4 члена в фикс. порядке ⇒
    перестановка при флипе даёт ~1e-6 ULP-шум), НО ошибка НЕСМЕЩЁННАЯ ⇒ не усиливается
    в асимметрию (в отличие от atan2, decisions.md R3). Проверяем малый разброс."""
    q = haar_quaternions(jax.random.PRNGKey(60), (10,))
    g = _grad_elastic("cosserat_chordal", q)
    gj = _grad_elastic("cosserat_chordal", quat_mul(q, _J))
    assert np.allclose(np.asarray(gj), np.asarray(quat_mul(g, _J)), atol=1e-4)


@pytest.mark.xfail(reason="atan2-Cosserat float32-неэквивариантность градиента — "
                          "документированный дефект (decisions.md R3); рабочий режим cosserat_chordal")
def test_atan2_cosserat_grad_bitwise_equivariant():
    q = haar_quaternions(jax.random.PRNGKey(60), (10,))
    g = _grad_elastic("cosserat", q)
    gj = _grad_elastic("cosserat", quat_mul(q, _J))
    assert np.array_equal(np.asarray(gj), np.asarray(quat_mul(g, _J)))


def test_cosserat_chordal_separates_twist_and_bend():
    """chordal-Коссера: чистая скрутка → только k_t·(2z)², чистый изгиб → только изгиб.
    Для малого φ: 4z²=4sin²(φ/2)≈φ², 4(x²+y²)≈φ²."""
    phi = 0.4
    z_rot = jnp.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], dtype=jnp.float64)
    x_rot = jnp.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0], dtype=jnp.float64)
    ident = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    k_b, k_t = 2.0, 3.0
    e_tw = float(e_cosserat_chordal(jnp.stack([ident, z_rot]), k_b, k_t))
    e_bend = float(e_cosserat_chordal(jnp.stack([ident, x_rot]), k_b, k_t))
    assert np.allclose(e_tw, 4 * k_t * np.sin(phi / 2) ** 2, atol=1e-9)   # только скрутка
    assert np.allclose(e_bend, 4 * k_b * np.sin(phi / 2) ** 2, atol=1e-9)  # только изгиб


def test_cosserat_separates_twist_and_bend():
    """R3: чистая скрутка даёт только k_t·φ², чистый изгиб — только k_b·φ²."""
    phi = 0.5
    z_rot = jnp.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], dtype=jnp.float64)
    x_rot = jnp.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0], dtype=jnp.float64)
    ident = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    k_b, k_t = 2.0, 3.0
    e_tw = float(e_cosserat(jnp.stack([ident, z_rot]), k_b, k_t))
    e_bend = float(e_cosserat(jnp.stack([ident, x_rot]), k_b, k_t))
    assert np.allclose(e_tw, k_t * phi ** 2, atol=1e-6)     # только скрутка
    assert np.allclose(e_bend, k_b * phi ** 2, atol=1e-6)   # только изгиб


def test_cosserat_grad_matches_finite_differences():
    """jax.grad Коссера против конечных разностей (SPEC §7, R3)."""
    N = 5
    q = haar_quaternions(jax.random.PRNGKey(33), (N,)).astype(jnp.float64)
    a = _unit(jnp.array([0.1, 0.2, 0.97], dtype=jnp.float64))
    b = _unit(jnp.array([0.8, -0.1, 0.4], dtype=jnp.float64))
    k_c, k_b, k_t = 1.1, 2.0, 0.5
    g = jax.grad(e_total)(q, a, b, 0.0, k_c, False, "cosserat", k_b, k_t)
    h = 1e-6
    fd = np.zeros_like(np.asarray(q)); qn = np.asarray(q)
    for i in range(N):
        for j in range(4):
            qp = qn.copy(); qp[i, j] += h
            qm = qn.copy(); qm[i, j] -= h
            ep = float(e_total(jnp.array(qp), a, b, 0.0, k_c, False, "cosserat", k_b, k_t))
            em = float(e_total(jnp.array(qm), a, b, 0.0, k_c, False, "cosserat", k_b, k_t))
            fd[i, j] = (ep - em) / (2 * h)
    assert np.allclose(np.asarray(g), fd, rtol=1e-4, atol=1e-6)


def test_spinor_elastic_penalizes_antipode():
    """elastic='spinor' (R5): d(q,−q)=π штрафуется, а geodesic (модуль) — нет."""
    q = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    chain = jnp.stack([q, -q])  # соседи-антиподы = кинк
    assert float(e_elastic(chain, 1.0, elastic="geodesic")) < 1e-4
    assert np.allclose(float(e_elastic(chain, 1.0, elastic="spinor")),
                       np.pi ** 2, atol=2e-2)


def test_spinor_grad_matches_finite_differences():
    N = 5
    q = haar_quaternions(jax.random.PRNGKey(32), (N,)).astype(jnp.float64)
    a = _unit(jnp.array([0.1, 0.2, 0.97], dtype=jnp.float64))
    b = _unit(jnp.array([0.8, -0.1, 0.4], dtype=jnp.float64))
    k_e, k_c = 0.7, 1.1
    g = jax.grad(e_total)(q, a, b, k_e, k_c, False, "spinor")
    h = 1e-6
    fd = np.zeros_like(np.asarray(q)); qn = np.asarray(q)
    for i in range(N):
        for j in range(4):
            qp = qn.copy(); qp[i, j] += h
            qm = qn.copy(); qm[i, j] -= h
            ep = float(e_total(jnp.array(qp), a, b, k_e, k_c, False, "spinor"))
            em = float(e_total(jnp.array(qm), a, b, k_e, k_c, False, "spinor"))
            fd[i, j] = (ep - em) / (2 * h)
    assert np.allclose(np.asarray(g), fd, rtol=1e-4, atol=1e-6)


def test_chordal_grad_matches_finite_differences():
    """jax.grad для chordal-режима против конечных разностей (SPEC §7, фаза C)."""
    N = 5
    q = haar_quaternions(jax.random.PRNGKey(31), (N,)).astype(jnp.float64)
    a = _unit(jnp.array([0.1, 0.2, 0.97], dtype=jnp.float64))
    b = _unit(jnp.array([0.8, -0.1, 0.4], dtype=jnp.float64))
    k_e, k_c = 0.7, 1.1
    g = jax.grad(e_total)(q, a, b, k_e, k_c, False, "chordal")
    h = 1e-6
    fd = np.zeros_like(np.asarray(q))
    qn = np.asarray(q)
    for i in range(N):
        for j in range(4):
            qp = qn.copy(); qp[i, j] += h
            qm = qn.copy(); qm[i, j] -= h
            ep = float(e_total(jnp.array(qp), a, b, k_e, k_c, False, "chordal"))
            em = float(e_total(jnp.array(qm), a, b, k_e, k_c, False, "chordal"))
            fd[i, j] = (ep - em) / (2 * h)
    assert np.allclose(np.asarray(g), fd, rtol=1e-4, atol=1e-6)
