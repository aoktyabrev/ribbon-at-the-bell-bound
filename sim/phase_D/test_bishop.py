"""Unit-тесты бишоповского транспорта SO(4) и канонического нормального базиса (D0 §1)."""

import jax.numpy as jnp
import jax.random as jr

import band4d as B
from ribbon_sim.frames import normalize


def _rand_unit(key):
    return normalize(jr.normal(key, (4,)))


def test_bishop_maps_ta_to_tb():
    """R_i t_i = t_{i+1}."""
    for seed in range(20):
        k = jr.PRNGKey(seed)
        ta, tb = _rand_unit(k), _rand_unit(jr.split(k)[1])
        got = B.bishop_apply(ta, tb, ta)
        assert float(jnp.linalg.norm(got - tb)) < 1e-12


def test_bishop_orthogonal_det_plus_one():
    """R ∈ SO(4): RᵀR=I и det=+1."""
    for seed in range(20):
        k = jr.PRNGKey(100 + seed)
        ta, tb = _rand_unit(k), _rand_unit(jr.split(k)[1])
        R = B.bishop_matrix(ta, tb)
        assert float(jnp.linalg.norm(R.T @ R - jnp.eye(4))) < 1e-12
        assert abs(float(jnp.linalg.det(R)) - 1.0) < 1e-10


def test_bishop_identity_when_equal():
    """t_i=t_{i+1} ⇒ R=I."""
    for seed in range(10):
        ta = _rand_unit(jr.PRNGKey(200 + seed))
        R = B.bishop_matrix(ta, ta)
        assert float(jnp.linalg.norm(R - jnp.eye(4))) < 1e-12


def test_bishop_identity_on_orthogonal_complement():
    """R тождественно на span(t_a,t_b)^⊥ (минимальность вращения)."""
    for seed in range(20):
        k = jr.PRNGKey(300 + seed)
        ta, tb = _rand_unit(k), _rand_unit(jr.split(k)[1])
        # вектор ⊥ ta и ⊥ tb (в дополнении плоскости)
        v = jr.normal(jr.PRNGKey(999 + seed), (4,))
        v = v - jnp.sum(v * ta) * ta
        tb_perp = tb - jnp.sum(tb * ta) * ta
        v = v - jnp.sum(v * tb_perp) / jnp.maximum(jnp.sum(tb_perp * tb_perp), 1e-12) * tb_perp
        v = normalize(v)
        Rv = B.bishop_apply(ta, tb, v)
        assert float(jnp.linalg.norm(Rv - v)) < 1e-10


def test_canonical_basis_orthonormal_perp_tangent():
    """P(t)=(t·i,t·j,t·k) — ОН-базис t^⊥."""
    for seed in range(20):
        t = _rand_unit(jr.PRNGKey(400 + seed))
        P = B.canonical_normal_basis(t)          # (3,4)
        assert float(jnp.linalg.norm(P @ P.T - jnp.eye(3))) < 1e-12
        assert float(jnp.linalg.norm(P @ t)) < 1e-12


def test_connection_identity_on_straight():
    """Прямой стержень ⇒ связь c_i = (1,0,0,0)."""
    x = jnp.zeros((10, 4)).at[:, 1].set(jnp.arange(10) * 1.0)
    c = B.connection_quats(B.node_tangents(x))
    assert float(jnp.max(jnp.abs(c - jnp.array([1.0, 0.0, 0.0, 0.0])))) < 1e-12
