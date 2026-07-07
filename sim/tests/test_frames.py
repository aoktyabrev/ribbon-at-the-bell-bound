"""Тесты геометрии кадров (SPEC §7)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ribbon_sim.frames import (
    axis,
    conj,
    geodesic,
    haar_quaternions,
    log_map,
    normalize,
    quat_mul,
    relative_log,
    rotmat,
)

IDENTITY = jnp.array([1.0, 0.0, 0.0, 0.0])
E_Z = jnp.array([0.0, 0.0, 1.0])


def _rand_quats(key, n):
    return haar_quaternions(key, (n,))


def test_normalize_unit_norm():
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (50, 4))
    qn = normalize(q)
    norms = jnp.linalg.norm(qn, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_axis_identity_is_ez():
    n = axis(IDENTITY)
    assert np.allclose(n, E_Z, atol=1e-6)


def test_rotmat_is_special_orthogonal():
    qs = _rand_quats(jax.random.PRNGKey(1), 20)
    R = rotmat(qs)
    # R Rᵀ = I и det = 1
    eye = jnp.einsum("bij,bkj->bik", R, R)
    dets = jnp.linalg.det(R)
    assert np.allclose(eye, jnp.eye(3)[None], atol=1e-5)
    assert np.allclose(dets, 1.0, atol=1e-5)


def test_rotmat_composition_matches_quat_mul():
    q1 = _rand_quats(jax.random.PRNGKey(2), 10)
    q2 = _rand_quats(jax.random.PRNGKey(3), 10)
    R12 = rotmat(quat_mul(q1, q2))
    RR = jnp.einsum("bij,bjk->bik", rotmat(q1), rotmat(q2))
    assert np.allclose(R12, RR, atol=1e-5)


def test_axis_matches_rotmat_third_column():
    qs = _rand_quats(jax.random.PRNGKey(4), 20)
    n = axis(qs)
    n_from_R = rotmat(qs)[..., :, 2]
    assert np.allclose(n, n_from_R, atol=1e-6)


def test_geodesic_zero_at_coincidence():
    qs = _rand_quats(jax.random.PRNGKey(5), 30)
    d = geodesic(qs, qs)
    # arccos обрезан на eps=1e-6 ⇒ d ≈ 0 (не строго 0), см. GEODESIC_EPS
    assert np.all(np.asarray(d) < 2e-3)


def test_geodesic_antipode_phase1_zero_spinor_pi():
    qs = _rand_quats(jax.random.PRNGKey(6), 30)
    # фаза 1 (модуль): d(q, −q) = 0
    d1 = geodesic(qs, -qs, spinor=False)
    assert np.all(np.asarray(d1) < 2e-3)
    # спинорный режим: d(q, −q) = π
    d2 = geodesic(qs, -qs, spinor=True)
    assert np.allclose(np.asarray(d2), np.pi, atol=2e-3)


def test_geodesic_known_angle():
    # поворот вокруг x на угол φ: q=(cos φ/2, sin φ/2,0,0); d(I,q)=φ/2 (фаза 1)
    phi = 1.0
    q = jnp.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])
    d = geodesic(IDENTITY, q, spinor=True)
    assert np.allclose(float(d), phi / 2, atol=1e-4)


def test_haar_uniform_mean_axis_is_zero():
    # Haar-мера изотропна: средняя ось ≈ 0
    qs = haar_quaternions(jax.random.PRNGKey(7), (200000,))
    mean_axis = jnp.mean(axis(qs), axis=0)
    assert np.all(np.abs(np.asarray(mean_axis)) < 1e-2)


def test_relative_log_pure_twist_and_bend():
    phi = 0.6
    z_rot = jnp.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)])  # поворот вокруг z
    x_rot = jnp.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])  # поворот вокруг x
    # цепочка [I, z_rot]: ω = (0,0,φ) — чистая скрутка
    chain_tw = jnp.stack([IDENTITY, z_rot])
    omega_tw = relative_log(chain_tw)[0]
    assert np.allclose(np.asarray(omega_tw), [0.0, 0.0, phi], atol=1e-5)
    # цепочка [I, x_rot]: ω = (φ,0,0) — чистый изгиб
    chain_bend = jnp.stack([IDENTITY, x_rot])
    omega_bend = relative_log(chain_bend)[0]
    assert np.allclose(np.asarray(omega_bend), [phi, 0.0, 0.0], atol=1e-5)


def test_relative_log_material_frame():
    # относительный поворот берётся в фрейме i: заранее повёрнутая цепочка даёт тот же ω
    phi = 0.5
    q0 = haar_quaternions(jax.random.PRNGKey(40), (1,))[0]
    z_rel = jnp.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)])
    q1 = quat_mul(q0, z_rel)  # q1 = q0 ∘ (твист вокруг локального z)
    omega = relative_log(jnp.stack([q0, q1]))[0]
    assert np.allclose(np.asarray(omega), [0.0, 0.0, phi], atol=1e-5)


def test_conj_is_inverse():
    qs = _rand_quats(jax.random.PRNGKey(41), 10)
    prod = quat_mul(conj(qs), qs)
    assert np.allclose(np.asarray(prod), IDENTITY, atol=1e-6)


def test_log_map_recovers_angle():
    phi = 0.7
    q = jnp.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])
    v = log_map(q)
    # |log| = физический угол поворота φ, направление вдоль x
    assert np.allclose(float(jnp.linalg.norm(v)), phi, atol=1e-5)
    assert np.allclose(np.asarray(v), [phi, 0.0, 0.0], atol=1e-5)
