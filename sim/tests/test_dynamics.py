"""Тесты динамики (SPEC §7): монотонность энергии при T=0, сохранение нормы,
батч-независимость."""

import jax
import jax.numpy as jnp
import numpy as np

from ribbon_sim.dynamics import branch_counts, build_relaxer, classify
from ribbon_sim.frames import haar_quaternions

BASE = {"N": 8, "k_e": 0.5, "k_c": 1.0, "spinor": False,
        "lr": 0.01, "T0": 0.0, "decay": 1.0, "steps": 300}


def _ab():
    a = jnp.array([0.0, 0.0, 1.0])
    b = jnp.array([np.sin(0.7), 0.0, np.cos(0.7)])
    return a, b


def test_energy_non_increasing_at_T0():
    cfg = dict(BASE)
    relax = build_relaxer(cfg)
    q0 = haar_quaternions(jax.random.PRNGKey(0), (32, cfg["N"]))
    a, b = _ab()
    _, e_trace = relax(jax.random.PRNGKey(1), q0, a, b)
    e = np.asarray(e_trace)
    # чистый градиентный спуск (T=0): энергия не растёт (с допуском на численность)
    diffs = np.diff(e)
    assert np.all(diffs <= 1e-4), f"максимальный рост энергии: {diffs.max():.2e}"
    assert e[-1] < e[0]  # и реально убыла


def test_quaternion_norm_preserved():
    cfg = dict(BASE, T0=0.2, decay=0.99)  # с шумом
    relax = build_relaxer(cfg)
    q0 = haar_quaternions(jax.random.PRNGKey(2), (16, cfg["N"]))
    a, b = _ab()
    qf, _ = relax(jax.random.PRNGKey(3), q0, a, b)
    norms = jnp.linalg.norm(qf, axis=-1)
    assert np.allclose(np.asarray(norms), 1.0, atol=1e-5)


def test_batch_independence():
    """Лента j не влияет на ленту k: изменение q0[0] не трогает qf[1:]."""
    cfg = dict(BASE, T0=0.15, decay=0.99)  # шум включён — проверяем реальную независимость
    relax = build_relaxer(cfg)
    a, b = _ab()
    key_noise = jax.random.PRNGKey(5)

    q0 = haar_quaternions(jax.random.PRNGKey(4), (12, cfg["N"]))
    qf_a, _ = relax(key_noise, q0, a, b)

    # заменяем ТОЛЬКО ленту 0 на другую начальную конфигурацию
    q0_mod = q0.at[0].set(haar_quaternions(jax.random.PRNGKey(99), (cfg["N"],)))
    qf_b, _ = relax(key_noise, q0_mod, a, b)

    # ленты 1..B-1 обязаны совпасть побитно (тот же шум, независимая динамика)
    assert np.allclose(np.asarray(qf_a[1:]), np.asarray(qf_b[1:]), atol=1e-6)
    # а лента 0 — измениться
    assert not np.allclose(np.asarray(qf_a[0]), np.asarray(qf_b[0]), atol=1e-3)


def test_classify_and_counts_shapes():
    cfg = dict(BASE)
    relax = build_relaxer(cfg)
    q0 = haar_quaternions(jax.random.PRNGKey(6), (64, cfg["N"]))
    a, b = _ab()
    qf, _ = relax(jax.random.PRNGKey(7), q0, a, b)
    s, t = classify(qf, a, b)
    assert set(np.unique(np.asarray(s))).issubset({-1, 1})
    counts = branch_counts(s, t)
    assert int(jnp.sum(counts)) == 64


def test_disconnected_gives_zero_correlation():
    """R0-логика в миниатюре: k_e=0 ⇒ E(θ)≈0, маргиналы ~0.5."""
    cfg = dict(BASE, k_e=0.0, N=16, steps=500)
    relax = build_relaxer(cfg)
    B = 8192
    q0 = haar_quaternions(jax.random.PRNGKey(8), (B, cfg["N"]))
    a, b = _ab()
    qf, _ = relax(jax.random.PRNGKey(9), q0, a, b)
    s, t = classify(qf, a, b)
    E = float(jnp.mean(s * t))
    p_s = float(jnp.mean(s > 0))
    assert abs(E) < 0.05
    assert abs(p_s - 0.5) < 0.05
