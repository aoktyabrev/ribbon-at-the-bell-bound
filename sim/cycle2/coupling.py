"""Цикл 2 — харнесс сопряжённых прогонов (C2-F) и пер-репличного поворота
приготовления (C2-ISO-DYN). См. prereg_drafts/C2F_*, C2ISO_*.

Существующий код фазы D НЕ трогается: build_minimizer/prep_dynamics/classify_*
импортируются как есть. Всё чистый JAX (jit/vmap-совместимо), идентификаторы
английские, комментарии русские. БЕЗ снятия физических оценок — только механика.

Два независимых блока:
  1. coupled_run — общий сырой драйв, разное b (C2-F). Ключ степпера не зависит
     от b (measurement.py:260) ⇒ при γ=0 выход b и b' ПОБИТОВО одинаков.
  2. haar_rotate_prep — жёсткий поворот приготовления R∈SO(3) на реплику
     (C2-ISO-DYN): позиции/границы поворачиваются в срезе Im ℍ, реперы левым
     умножением q_R⊗u0. Оси (a,b) остаются лаб-фиксированными.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from ribbon_sim.frames import normalize, quat_mul, rotmat
import measurement as M


# --------------------------------------------------------------------------- #
#  C2-F: сопряжённые прогоны (общий λ, разное b)
# --------------------------------------------------------------------------- #

def coupled_run(mini, prep, key, a, b, b_prime, T_schedule):
    """Пара прогонов при СОВПАДАЮЩЕЙ λ: тот же key, prep, расписание — различие
    ТОЛЬКО в удалённой оси b vs b_prime (a общая). Возврат (u_b, u_bp), формы (M,N,4).

    Общий key ⇒ идентичная раздача шума в степпере (b в split не входит,
    measurement.py:260). Оговорка prereg: побитово общий лишь СЫРОЙ драйв;
    эффективные приращения расходятся через тангенциальную проекцию."""
    run = mini["run"]
    _, u_b, _ = _unpack(run(key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, T_schedule))
    _, u_bp, _ = _unpack(run(key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b_prime, T_schedule))
    return u_b, u_bp


def _unpack(ret):
    """build_minimizer.run возвращает (x,u,rej); терпим и (x,u)."""
    if len(ret) == 3:
        return ret
    x, u = ret
    return x, u, None


def flip_rate(u_b, u_bp, a, b, b_prime):
    """F = P(s(λ,a,b) ≠ s(λ,a,b')) — flip-rate ближнего конца (s = знак на a).

    Знаки берём как в analysis_ds3.iso_corr: sign(0)→+1, БЕЗ отсечения degen
    (degen возвращается диагностикой). u_* формы (M,N,4)."""
    sb, _, deg_b = M.classify_batch(u_b, a, b)
    sbp, _, deg_bp = M.classify_batch(u_bp, a, b_prime)
    sb, sbp = jnp.asarray(sb), jnp.asarray(sbp)
    F = float(jnp.mean(sb != sbp))
    sigF = float(jnp.sqrt(max(F * (1 - F), 1e-9) / sb.size))
    degen_diag = float(jnp.mean(deg_b | deg_bp))
    return F, sigF, degen_diag


def marginal_gap(u_b, u_bp, a, b, b_prime):
    """Δ = |P(s=+|a,b) − P(s=+|a,b')| — сдвиг маргинала ближнего конца."""
    sb, _, _ = M.classify_batch(u_b, a, b)
    sbp, _, _ = M.classify_batch(u_bp, a, b_prime)
    pb = float(jnp.mean(jnp.asarray(sb) > 0))
    pbp = float(jnp.mean(jnp.asarray(sbp) > 0))
    return abs(pb - pbp), pb, pbp


def null_test_gamma0(mini, prep, key, a, b, T_schedule):
    """Нуль-тест пайплайна (C2-F prereg): γ=0 ⇒ b'=b ⇒ выходы ПОБИТОВО равны.
    Возврат (max_abs_diff, ok). Любое max_abs_diff≠0 — баг раздачи ключей."""
    u_b, u_bp = coupled_run(mini, prep, key, a, b, b, T_schedule)
    diff = float(jnp.max(jnp.abs(u_b - u_bp)))
    return diff, diff == 0.0


# --------------------------------------------------------------------------- #
#  C2-ISO-DYN: пер-репличный жёсткий поворот приготовления
# --------------------------------------------------------------------------- #

def haar_quats(key, m):
    """M равномерных (Haar) кватернионов на S³ — представители SO(3).
    Алгоритм Shoemake. Возврат (M,4), w≥0 (короткая ветвь; на осевую
    статистику знак лифта не влияет)."""
    u1, u2, u3 = jr.uniform(key, (3, m))
    q = jnp.stack([
        jnp.sqrt(1 - u1) * jnp.sin(2 * jnp.pi * u2),
        jnp.sqrt(1 - u1) * jnp.cos(2 * jnp.pi * u2),
        jnp.sqrt(u1) * jnp.sin(2 * jnp.pi * u3),
        jnp.sqrt(u1) * jnp.cos(2 * jnp.pi * u3),
    ], axis=-1)                                     # (M,4), порядок (x,y,z,w)-ish
    q = jnp.concatenate([q[:, 3:4], q[:, :3]], axis=1)   # → (w,x,y,z)
    return q * jnp.sign(q[:, 0:1] + 1e-30)


def _rotate_slice(x, R):
    """Поворот пер-репличной матрицей R (M,3,3) точек в срезе Im ℍ.
    x (M,...,4): w-компонента (ось 0) нетронута, компоненты 1:4 → R·v."""
    w = x[..., :1]
    v = x[..., 1:]
    # einsum по ведущей оси реплики M; промежуточные оси броадкастятся
    if v.ndim == 3:      # (M,N,3)
        v = jnp.einsum("mij,mnj->mni", R, v)
    else:                # (M,3)
        v = jnp.einsum("mij,mj->mi", R, v)
    return jnp.concatenate([w, v], axis=-1)


def haar_rotate_prep(prep, q_R):
    """Жёсткий поворот ВСЕГО приготовления на реплику (C2-ISO-DYN prereg).

    q_R (M,4) — Haar-кватернионы. Позиции x0 и границы X0,XL поворачиваются
    в срезе Im ℍ матрицей R(q_R); реперы u0 ← q_R ⊗ u0 (левое умножение).
    Оси (a,b) НЕ трогаются — они лаб-фиксированы (в этом суть протокола).

    Инвариантность e_meas под (x,u,оси)→(RxR⁻¹, q⊗u, R·оси) проверена
    юнит-зондом (Δ<1e-14); здесь оси намеренно фиксированы ⇒ поворот
    зондирует анизотропию A(α) под Haar-мерой. Возврат новый dict prep."""
    R = rotmat(q_R)                                  # (M,3,3)
    x0 = _rotate_slice(prep["x0"], R)
    X0 = _rotate_slice(prep["X0"], R)
    XL = _rotate_slice(prep["XL"], R)
    u0 = normalize(quat_mul(q_R[:, None, :], prep["u0"]))
    return dict(prep, x0=x0, u0=u0, X0=X0, XL=XL)
