"""Тесты параллельного темперирования (R4b-PT).

Ядро обмена тестируется на игрушечной двухъямке (Euclid), независимо от S³.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ribbon_sim.pt import geometric_ladder, make_pt


def _double_well(tilt=0.0):
    """V(x) = (x²−1)² + tilt·x. Ямы у x≈±1, барьер ~1 у x=0."""
    def energy(state):            # state (B,R,1) → (B,R)
        x = state[..., 0]
        return (x * x - 1.0) ** 2 + tilt * x
    grad = jax.grad(lambda s: jnp.sum((s[..., 0] ** 2 - 1.0) ** 2 + tilt * s[..., 0]))

    def step(state, Ts, key):     # Euclid-ланжевен, Ts形 (R,)
        lr = 0.01
        g = grad(state)
        noise = jax.random.normal(key, state.shape) * jnp.sqrt(2 * lr * Ts)[None, :, None]
        return state - lr * g + noise
    return energy, step


def test_pt_crosses_barrier_symmetric():
    """Симметричная двухъямка: PT населяет ОБЕ ямы ~50/50 на холодной реплике,
    тогда как одиночная холодная динамика застряла бы в одной (детальный баланс/эргодичность)."""
    energy, step = _double_well(tilt=0.0)
    betas = 1.0 / geometric_ladder(2.0, 0.05, 8)  # T от 2 до 0.05
    run = make_pt(energy, step, betas, swap_every=20)
    B = 4096
    key = jax.random.PRNGKey(0)
    # старт: все в ЛЕВОЙ яме (x=−1) — проверяем, что PT населит и правую
    state0 = -jnp.ones((B, 8, 1)) + 0.01 * jax.random.normal(key, (B, 8, 1))
    state, diag = run(jax.random.PRNGKey(1), state0, n_blocks=400)
    x_cold = np.asarray(state[:, -1, 0])  # холодная реплика = последняя (T_cold)
    frac_right = float(np.mean(x_cold > 0))
    assert 0.4 < frac_right < 0.6, f"ямы не 50/50: P(right)={frac_right:.2f}"
    # ⟨x²⟩ ≈ 1 (ямы у ±1) при низкой T
    assert abs(float(np.mean(x_cold ** 2)) - 1.0) < 0.15


def test_pt_boltzmann_ratio_tilted():
    """Наклонённая двухъямка: отношение населённостей ям ≈ exp(−ΔV/T_cold) (больцман)."""
    tilt = 0.3
    energy, step = _double_well(tilt=tilt)
    T_cold = 0.15
    betas = 1.0 / geometric_ladder(2.0, T_cold, 8)
    run = make_pt(energy, step, betas, swap_every=20)
    B = 8192
    state0 = 0.01 * jax.random.normal(jax.random.PRNGKey(2), (B, 8, 1))
    state, _ = run(jax.random.PRNGKey(3), state0, n_blocks=500)
    x = np.asarray(state[:, -1, 0])
    # минимумы наклонённой ямы: V'(x)=4x(x²−1)+tilt=0. Численно найдём два корня у ±1.
    from numpy.polynomial import polynomial as P  # noqa
    roots = np.roots([4, 0, -4, tilt])
    real = np.sort(roots[np.abs(roots.imag) < 1e-6].real)
    xL, xR = real[0], real[-1]
    VL = (xL ** 2 - 1) ** 2 + tilt * xL
    VR = (xR ** 2 - 1) ** 2 + tilt * xR
    ratio_theory = np.exp(-(VL - VR) / T_cold)     # P(L)/P(R)
    nL = np.mean(x < 0); nR = np.mean(x > 0)
    ratio_emp = nL / max(nR, 1e-6)
    # порядок величины (гармонические префакторы не учтены) — допускаем фактор 2
    assert 0.5 < ratio_emp / ratio_theory < 2.0, \
        f"больцман: эмп={ratio_emp:.2f} теор={ratio_theory:.2f}"


def test_pt_conserves_twist_ribbons():
    """R4b-PT: обмены реплик сохраняют Tw ленты (все реплики в общем секторе)."""
    from ribbon_sim.dynamics import build_pt_fns
    from ribbon_sim.frames import sector_sample, total_twist
    from ribbon_sim.pt import make_pt

    N, B, R = 32, 128, 6
    tw_target = 2 * np.pi
    q0_flat, _ = sector_sample(jax.random.PRNGKey(6), (B, N), target=tw_target)
    # реплики: копии одной ленты (общий сектор)
    state0 = jnp.broadcast_to(q0_flat[:, None], (B, R, N, 4))
    a = jnp.array([0.0, 0.0, 1.0]); b = jnp.array([np.sin(0.7), 0.0, np.cos(0.7)])
    cfg = {"lr": 0.005, "k_e": 0.0, "k_c": 1.0, "spinor": True, "elastic": "cosserat_geo",
           "k_b": 90.0, "k_t": 9.0, "twist_project": True}
    energy_fn, step_fn = build_pt_fns(cfg, a, b, tw_target)
    betas = 1.0 / geometric_ladder(1.0, 1e-3, R)
    run = make_pt(energy_fn, step_fn, betas, swap_every=20)
    state, _ = run(jax.random.PRNGKey(7), state0, n_blocks=30)
    tw = np.asarray(jax.vmap(jax.vmap(total_twist))(state))  # (B,R)
    # СЕКТОР сохранён: |ΔTw| ≪ π (ℤ₂-класс не сменился на 0 или 4π). Итерированная
    # коррекция держит дрейф на ~0.01–0.03; свопы точны (перемаркировка внутри сектора).
    assert np.max(np.abs(tw - tw_target)) < 0.1, f"Tw дрейф {np.max(np.abs(tw-tw_target)):.3f}"


def test_pt_swaps_exact_within_sector():
    """Свопы реплик точно сохраняют Tw (обмен конфигураций внутри общего сектора ленты):
    без ланжевен-дрейфа (T→0 у всех) обмены не меняют множество Tw реплик ленты."""
    from ribbon_sim.dynamics import build_pt_fns
    from ribbon_sim.frames import sector_sample, total_twist
    from ribbon_sim.pt import make_pt

    N, B, R = 16, 64, 6
    tw_target = 2 * np.pi
    q0_flat, _ = sector_sample(jax.random.PRNGKey(8), (B, N), target=tw_target)
    state0 = jnp.broadcast_to(q0_flat[:, None], (B, R, N, 4))
    a = jnp.array([0.0, 0.0, 1.0]); b = jnp.array([0.0, 0.0, 1.0])
    cfg = {"lr": 1e-9, "k_e": 0.0, "k_c": 1.0, "spinor": True, "elastic": "cosserat_geo",
           "k_b": 90.0, "k_t": 9.0, "twist_project": True, "n_twist_corr": 20}
    energy_fn, step_fn = build_pt_fns(cfg, a, b, tw_target)
    betas = 1.0 / geometric_ladder(1.0, 1e-3, R)  # разные T, но lr→0 ⇒ движения нет
    run = make_pt(energy_fn, step_fn, betas, swap_every=5)
    state, _ = run(jax.random.PRNGKey(9), state0, n_blocks=20)
    tw = np.asarray(jax.vmap(jax.vmap(total_twist))(state))
    assert np.max(np.abs(tw - tw_target)) < 1e-3, "свопы/пустой шаг сместили Tw"


def test_pt_acceptance_and_roundtrips_diagnostics():
    """Диагностика PT: acceptance по звеньям в (0,1], round-trips ≥ 0 и растут со временем."""
    energy, step = _double_well()
    betas = 1.0 / geometric_ladder(2.0, 0.1, 10)
    run = make_pt(energy, step, betas, swap_every=20)
    state0 = 0.01 * jax.random.normal(jax.random.PRNGKey(4), (2048, 10, 1))
    _, diag = run(jax.random.PRNGKey(5), state0, n_blocks=600)
    acc = np.asarray(diag["accept_frac"])
    assert acc.shape == (9,)
    assert np.all(acc >= 0) and np.all(acc <= 1)
    assert np.all(acc > 0.05), f"звенья без обменов: {acc}"
    assert int(diag["min_roundtrips"]) >= 0
    assert float(diag["mean_roundtrips"]) > 0  # какие-то уокеры прошли лестницу
