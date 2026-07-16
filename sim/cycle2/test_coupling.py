"""Цикл 2 — smoke-тесты харнеса (M≤8): ИСПОЛНЯЕМОСТЬ и ИНВАРИАНТЫ, БЕЗ снятия
физических оценок (F/ρ/E со smoke не читаются — это подглядывание до prereg).

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D:cycle2 pytest cycle2/test_coupling.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M
import coupling as C

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
N, MB, STEPS = 16, 8, 12          # smoke: крошечные, только механика


def _mini():
    return M.build_minimizer(BASE, lr=5e-3, freeze_w=False)


def _prep(key, seed_off=0):
    return M.prep_dynamics(N, True, MB, jr.fold_in(key, seed_off))


def test_null_gamma0_bit_identical():
    """Нуль-тест C2-F: γ=0 ⇒ выход b и b'=b ПОБИТОВО равны (общий key степпера)."""
    key = jr.PRNGKey(1)
    mini, prep = _mini(), _prep(key)
    a, b = M.apparatus_axes_theta(0.0)
    T = jnp.full((STEPS,), 0.05)
    diff, ok = C.null_test_gamma0(mini, prep, key, a, b, T)
    assert ok and diff == 0.0, f"нуль-тест провален: max|Δ|={diff}"


def test_coupled_run_finite_and_shaped():
    """coupled_run исполняется, выходы конечны и правильной формы (M,N,4)."""
    key = jr.PRNGKey(2)
    mini, prep = _mini(), _prep(key)
    a, b = M.apparatus_axes_theta(0.0)
    _, bp = M.apparatus_axes_theta(np.pi / 4)
    T = jnp.full((STEPS,), 0.0)     # квенч T=0
    u_b, u_bp = C.coupled_run(mini, prep, key, a, b, bp, T)
    assert u_b.shape == (MB, N, 4) and u_bp.shape == (MB, N, 4)
    assert jnp.all(jnp.isfinite(u_b)) and jnp.all(jnp.isfinite(u_bp))
    F, sigF, deg = C.flip_rate(u_b, u_bp, a, b, bp)
    assert 0.0 <= F <= 1.0 and np.isfinite(sigF) and 0.0 <= deg <= 1.0  # конечность, НЕ оценка


def test_haar_quats_unit_and_wnonneg():
    """Haar-кватернионы: единичная норма, w≥0 (короткая ветвь)."""
    q = C.haar_quats(jr.PRNGKey(3), MB)
    assert q.shape == (MB, 4)
    assert jnp.allclose(jnp.linalg.norm(q, axis=-1), 1.0, atol=1e-6)
    assert jnp.all(q[:, 0] >= -1e-12)


def test_tcov1_full_rotation_invariant():
    """T-cov-1: поворот И приготовления, И осей ОДНИМ R ⇒ e_meas инвариантна
    побитово (в пределах fp64). Проверяет корректность прескрипции поворота."""
    key = jr.PRNGKey(4)
    prep = _prep(key)
    q = C.haar_quats(jr.PRNGKey(40), MB)
    R = jax.vmap(lambda qq: __import__("ribbon_sim.frames", fromlist=["rotmat"]).rotmat(qq))(q)
    prep_rot = C.haar_rotate_prep(prep, q)
    a, b = M.apparatus_axes_theta(0.7)
    e_ref = jax.vmap(lambda x, u: M.e_meas(x, u, a, b, BASE))(prep["x0"], prep["u0"])
    # оси поворачиваем ТЕМ ЖЕ R на реплику ⇒ должно совпасть
    e_rot = jax.vmap(lambda x, u, Rm: M.e_meas(x, u, Rm @ a, Rm @ b, BASE))(
        prep_rot["x0"], prep_rot["u0"], R)
    assert jnp.allclose(e_ref, e_rot, atol=1e-9), \
        f"T-cov-1: max|Δe|={float(jnp.max(jnp.abs(e_ref - e_rot))):.2e}"


def test_tcov2_inner_axis_noop():
    """T-cov-2: поворот приготовления вокруг ВНУТРЕННЕЙ оси ê (считывающей)
    при лаб-фиксированных осях — осевые проекции |n·a| статистически те же
    (энергия зажима −k_c(n·a)² инвариантна к вращению репера вокруг a).
    Smoke: проверяем ИСПОЛНЯЕМОСТЬ и конечность, не распределение."""
    key = jr.PRNGKey(5)
    prep = _prep(key)
    # поворот вокруг ê=(0,0,1) на случайный угол на реплику
    ang = jr.uniform(jr.PRNGKey(50), (MB,)) * 2 * np.pi
    half = 0.5 * ang
    z = jnp.zeros(MB)
    q_ez = jnp.stack([jnp.cos(half), z, z, jnp.sin(half)], axis=-1)  # exp(½·ang·ẑ)
    prep_rot = C.haar_rotate_prep(prep, q_ez)
    assert jnp.all(jnp.isfinite(prep_rot["x0"])) and jnp.all(jnp.isfinite(prep_rot["u0"]))
    assert jnp.allclose(jnp.linalg.norm(prep_rot["u0"], axis=-1), 1.0, atol=1e-6)


def test_identity_rotation_noop():
    """Тождественный поворот (q=identity): позиции/границы — ПОБИТОВО те же
    (матричный путь, R=I точно); реперы — до fp64-округления (normalize
    делит на норму ⇒ 1 ULP). Именно это гарантия «новый путь не искажает».

    Истинная регрессия старого кода (build_minimizer/prep_dynamics побитово
    неизменны) — факт уровня git: cycle2/ только ДОБАВЛЯЕТ файлы, phase_D не
    трогается. Здесь — свойство самого поворота."""
    key = jr.PRNGKey(6)
    prep = _prep(key)
    q_id = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (MB, 1))
    pr = C.haar_rotate_prep(prep, q_id)
    for k in ("x0", "X0", "XL"):
        d = float(jnp.max(jnp.abs(pr[k] - prep[k])))
        assert d == 0.0, f"{k} должен быть побитово равен: max|Δ|={d}"
    du = float(jnp.max(jnp.abs(pr["u0"] - prep["u0"])))
    assert du < 1e-15, f"u0 сверх fp64-округления: max|Δ|={du}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("все smoke-тесты пройдены")
