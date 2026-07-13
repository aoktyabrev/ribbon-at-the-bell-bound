"""Фаза D, стадия D0 — оснащённая кривая в R⁴ (см. D0_prereg.md, D0_prereg_addendum.md).

Модель: кривая x_i ∈ R⁴ (i=0..N−1) + оснащение u_i ∈ S³ — лифт SO(3)-вращения
нормального трёхгранника относительно КАНОНИЧЕСКОГО бишоповского репера.

Ключевая геометрия (аддендум §1): R⁴ ≡ ℍ (кватернионы, скаляр-первый (w,x,y,z)).
S³ параллелизуема ⇒ для единичного касательного t канонический ортонормальный
базис его нормального 3-пространства t^⊥ берётся правым сдвигом мнимых единиц:
    P(t) = ( t⊗i, t⊗j, t⊗k ),   i=(0,1,0,0), j=(0,0,1,0), k=(0,0,0,1).
Это ГЛАДКИЙ (без «волосатого шара») базис t^⊥ по всей S³. Срез R³ = Im ℍ
(скалярная компонента w = index 0 — «лишнее» 4-е измерение); «зажать w=0» =
удержать позиции чисто мнимыми (component[...,0]=0).

Всё — чистый JAX. Функции на ОДНОЙ кривой; батч даётся vmap в скриптах прогонов.
Идентификаторы английские, формулы со ссылкой на аддендум (# ADD §k).
"""

import jax
import jax.numpy as jnp

from ribbon_sim.frames import conj, geodesic, normalize, quat_mul

# Мнимые единицы ℍ как кватернионы (скаляр-первый).
_IMAG = jnp.array(
    [[0.0, 1.0, 0.0, 0.0],   # i
     [0.0, 0.0, 1.0, 0.0],   # j
     [0.0, 0.0, 0.0, 1.0]],  # k
)

# Пороги отбраковки сингулярных шагов (ADD §4). Переопределяемы из конфига прогона.
DELTA_SING = 2.0e-2   # шаг по u отклоняется, если <u_new,u_old> < 1−δ_sing
DELTA_TAN = 2.0e-2    # шаг отклоняется, если t_i·t_{i+1} < −1+δ_tan (реверс касательной)


# --------------------------------------------------------------------------- #
#  Геометрия: касательные, канонический нормальный базис
# --------------------------------------------------------------------------- #

def node_tangents(x):
    """Пер-узловые касательные t_i ∈ S³ (ADD §1). x (N,4) → t (N,4).

    t_i = normalize(x_{i+1}−x_i) для i=0..N−2; t_{N−1} := t_{N−2} (последний узел
    наследует последнее ребро). Так число касательных = число реперов = N.
    """
    e = x[1:] - x[:-1]                      # (N−1, 4) рёбра
    t = normalize(e)                        # (N−1, 4)
    return jnp.concatenate([t, t[-1:]], axis=0)   # (N, 4)


def canonical_normal_basis(t):
    """P(t) = (t⊗i, t⊗j, t⊗k) — ортонормальный базис t^⊥ ⊂ R⁴ (ADD §1).

    t (..., 4) → P (..., 3, 4): три вектора-строки. Гладкий по t (параллелизуемость S³).
    """
    # broadcast: (...,1,4) ⊗ (3,4) → (...,3,4) правым сдвигом мнимых единиц
    return quat_mul(t[..., None, :], _IMAG)


# --------------------------------------------------------------------------- #
#  Бишоповский транспорт SO(4) через двойное отражение (ADD §2)
# --------------------------------------------------------------------------- #

def _householder(v, n):
    """Отражение H(n) v = v − 2⟨n,v⟩ n относительно гиперплоскости ⊥ n (|n|=1).
    v, n формы (..., 4)."""
    coef = 2.0 * jnp.sum(n * v, axis=-1, keepdims=True)
    return v - coef * n


def bishop_apply(t_a, t_b, v):
    """Применить минимальное вращение R (t_a→t_b) к вектору(ам) v (ADD §2).

    R = H(m)·H(t_a), m = normalize(t_a+t_b): двойное отражение. Тождественно на
    span(t_a,t_b)^⊥, R t_a = t_b, det=+1. При t_a=t_b даёт тождество.
    t_a,t_b формы (...,4); v формы (...,4) (broadcast по ведущим осям).
    Защита от t_a=−t_b (отбраковывается δ_tan, но grad/jit не должны NaN-ить):
    знаменатель |t_a+t_b| подпёрт снизу.
    """
    s = t_a + t_b
    norm = jnp.linalg.norm(s, axis=-1, keepdims=True)
    m = s / jnp.maximum(norm, 1e-12)
    w = _householder(v, t_a)
    return _householder(w, m)


def bishop_matrix(t_a, t_b):
    """Матрица R ∈ SO(4) минимального вращения t_a→t_b (для unit-тестов).
    t_a,t_b формы (4,) → R (4,4), столбцы R = образы базиса. R e_k = bishop_apply(e_k)."""
    eye = jnp.eye(4)
    cols = bishop_apply(t_a[None, :], t_b[None, :], eye)   # (4,4): строка k = R e_k
    return cols.T                                          # столбцы = образы


# --------------------------------------------------------------------------- #
#  Геометрическая связь c_i (ADD §3): лифт SO(3)-вращения, связывающего
#  бишоп-транспортированный канонический базис с каноническим базисом следующего узла
# --------------------------------------------------------------------------- #

def _mat3_to_quat(C):
    """Лифт SO(3)-матрицы C (3,3) → единичный кватернион (w,x,y,z), ветвь w≥0.

    C по построению — МАЛОЕ вращение (близкие касательные, δ_tan-защита) ⇒ trace≈3>0,
    берём устойчивую ветвь Шеппарда trace>0 (без ветвлений; безопасна у тождества).
    """
    tr = C[..., 0, 0] + C[..., 1, 1] + C[..., 2, 2]
    w = 0.5 * jnp.sqrt(jnp.maximum(1.0 + tr, 1e-12))
    d = 1.0 / (4.0 * w)
    x = (C[..., 2, 1] - C[..., 1, 2]) * d
    y = (C[..., 0, 2] - C[..., 2, 0]) * d
    z = (C[..., 1, 0] - C[..., 0, 1]) * d
    return jnp.stack([w, x, y, z], axis=-1)


def connection_quats(t):
    """Связь c_i (ADD §3), i=0..N−2. t (N,4) → c (N−1,4).

    P_i = canonical_normal_basis(t_i); T_i = R_i P_i (бишоп-транспорт в t_{i+1}^⊥);
    C_i = T_iᵀ P_{i+1} — матрица перекрытия 3×3 (оба — ОН-базисы t_{i+1}^⊥) ∈ SO(3);
    c_i = lift(C_i). При t_i=t_{i+1}: R_i=I, T_i=P_{i+1} ⇒ C_i=I ⇒ c_i=(1,0,0,0).
    """
    t_a, t_b = t[:-1], t[1:]                       # (N−1, 4)
    P_a = canonical_normal_basis(t_a)              # (N−1, 3, 4)
    P_b = canonical_normal_basis(t_b)              # (N−1, 3, 4)
    # T_i = R_i P_i: применяем bishop к каждому из 3 базисных векторов
    T = bishop_apply(t_a[:, None, :], t_b[:, None, :], P_a)   # (N−1, 3, 4)
    # C_i[a,b] = <T_i[a], P_b[b]>  → (N−1, 3, 3)
    C = jnp.einsum("iak,ibk->iab", T, P_b)
    return _mat3_to_quat(C)                         # (N−1, 4)


def twist_angles(u, t):
    """Локальные углы твиста φ_i ∈ [0,π] на связь: φ_i = 2·atan2(|g_i.vec|, |g_i.w|),
    геодезический угол SO(3)-приращения g_i (ADD §3). u,t (N,4) → φ (N−1,).
    Наблюдаемая для гистограмм/KS-теста локальной скрутки (V1)."""
    g = twist_increments(u, t)
    vn = jnp.linalg.norm(g[..., 1:], axis=-1)
    return 2.0 * jnp.arctan2(vn, jnp.abs(g[..., 0]))


def twist_increments(u, t):
    """Внутренние приращения твиста g_i = u_i⁻¹ ⊗ c_i ⊗ u_{i+1} (ADD §3), i=0..N−2.

    u (N,4), t (N,4) → g (N−1,4). g_i — SO(3)-вращение оснащения БЕЗ геометрии
    (связь c_i вычтена): g_i=identity ⇔ оснащение параллельно (бишоп-транспорт).
    """
    c = connection_quats(t)                        # (N−1, 4)
    return quat_mul(quat_mul(conj(u[:-1]), c), u[1:])


# --------------------------------------------------------------------------- #
#  SO(2)-ограниченное оснащение (ADD §8): одна DOF φ_i на узел
# --------------------------------------------------------------------------- #

def u_of_phi(phi):
    """SO(2)-репер u_i(φ_i)=exp(½φ_i ẑ)=(cos φ_i/2, 0, 0, sin φ_i/2), ẑ=(0,0,0,1).
    phi (...,N) → u (...,N,4). Вращение трёхгранника вокруг касательно-сопутствующей
    нормали (ковариантно-постоянный твист-DOF, ADD §8.2)."""
    half = 0.5 * phi
    z = jnp.zeros_like(phi)
    return jnp.stack([jnp.cos(half), z, z, jnp.sin(half)], axis=-1)


def winding(phi):
    """winding W = (1/2π)·Σ_i Δφ_i (ADD §8.3). φ хранится НЕПРЕРЫВНО (unwrapped) ⇒
    телескоп: W=(φ_{N−1}−φ_0)/2π. phi (...,N) → (...,)."""
    return (phi[..., -1] - phi[..., 0]) / (2.0 * jnp.pi)


def local_twist_so2(phi):
    """Локальная скрутка Δφ_i (ADD §8.3b) для гистограммы. phi (...,N) → (...,N−1)."""
    return phi[..., 1:] - phi[..., :-1]


# --------------------------------------------------------------------------- #
#  Энергия (ADD §5)
# --------------------------------------------------------------------------- #

def e_stretch(x, k_s, ell):
    """k_s Σ (|x_{i+1}−x_i| − ℓ)²."""
    d = jnp.linalg.norm(x[1:] - x[:-1], axis=-1)
    return k_s * jnp.sum((d - ell) ** 2)


def e_bend(t, k_b):
    """k_b Σ (1 − t_i·t_{i+1}), i=0..N−2."""
    dots = jnp.sum(t[:-1] * t[1:], axis=-1)
    return k_b * jnp.sum(1.0 - dots)


def e_twist(u, t, k_f):
    """k_f Σ d²(u_i,u_{i+1}) «с учётом транспорта» (ADD §5):
    d = geodesic на S³ от приращения g_i к тождеству = arccos(|g_i.w|).
    Изотропная (короткая) метрика — знак твиста несёт parity, не энергия."""
    g = twist_increments(u, t)                     # (N−1, 4)
    d = geodesic(g, jnp.array([1.0, 0.0, 0.0, 0.0]), spinor=False)   # arccos(|g.w|)
    return k_f * jnp.sum(d * d)


def e_clamp_frame(u_end, U_target, k_c):
    """Мягкий зажим граничного репера: k_c · d²(u_end, U_target), НЕспинорная геодезия
    arccos(|<·>|) (ADD §5). ±-СИММЕТРИЧНЫЙ (как −k_c(n·axis)² фаз A–C): аппаратура
    удерживает SO(3)-ОРИЕНТАЦИЮ репера и НЕ различает лифт u/−u. Сектор (чётность)
    задаётся ТОПОЛОГИЕЙ (фикс. реперы-ссылки в parity) и защищён стеной лифта, а НЕ
    энергией зажима — иначе зажим тянул бы нечётный сектор (u_end=−1) раскрутиться."""
    d = geodesic(u_end, U_target, spinor=False)
    return k_c * d * d


def e_total(x, u, U_A, U_B, params):
    """Полная энергия одной кривой (ADD §5). params — dict k_s,k_b,k_f,k_c,ell.
    Чистая от СЫРЫХ (x,u) — так grad и конечные разности видят одну функцию."""
    t = node_tangents(x)
    return (
        e_stretch(x, params["k_s"], params["ell"])
        + e_bend(t, params["k_b"])
        + e_twist(u, t, params["k_f"])
        + e_clamp_frame(u[0], U_A, params["k_c"])
        + e_clamp_frame(u[-1], U_B, params["k_c"])
    )


# --------------------------------------------------------------------------- #
#  Ланжевен-динамика на (x,u) с отбраковкой сингулярностей (ADD §4)
# --------------------------------------------------------------------------- #

def build_stepper(params, freeze_w=False, frame_group="S3",
                  delta_sing=DELTA_SING, delta_tan=DELTA_TAN, wind_tol=0.01):
    """Собирает jit-шаг батч-релаксации (ADD §4, §8).

    frame_group:
      - "S3"  — полное оснащение u_i ∈ S³ (R⁴-ветка, по умолчанию);
      - "SO2" — контроль V1-C: u_i=exp(½φ_i ẑ), одна DOF φ_i на узел (ADD §8.2).

    Конвенции ланжевена из фаз A–C: x += −lr∇E + √(2·lr·T)·ξ. Граничные УЗЛЫ: позиции
    x_0,x_{N−1} жёстко фиксированы (даны в X0/XL, w=0); реперы свободны (мягкий зажим).
    freeze_w=True — R³-срез: скалярная (w) компонента всех позиций обнулена.

    S3-режим: шаг по u — касательный + ренормировка; знаковая непрерывность
    (sign u_i(t+dt) ближайший к u_i(t)); отбраковка при <u_new,u_old><1−δ_sing или
    t_i·t_{i+1}<−1+δ_tan. run(key,x0,u0,X0,XL,U_A,U_B,T_sched)→(x,u,rejected).

    SO2-режим (ADD §8.6): шаг по φ; тангенциальная отбраковка сохраняется, лифтовая
    заменяется контролем winding — шаг с |W−W_target|≥wind_tol идёт в rejected.
    run_so2(key,x0,phi0,X0,XL,U_A,U_B,T_sched,W_target)→(x,phi,rejected).
    """
    lr = float(params["lr"])
    k_sing = 1.0 - float(delta_sing)
    k_tan = -1.0 + float(delta_tan)

    def e1(x, u, U_A, U_B):
        return e_total(x, u, U_A, U_B, params)

    grad_x = jax.grad(e1, argnums=0)
    grad_u = jax.grad(e1, argnums=1)
    gx_b = jax.vmap(grad_x, in_axes=(0, 0, 0, 0))
    gu_b = jax.vmap(grad_u, in_axes=(0, 0, 0, 0))
    tan_b = jax.vmap(node_tangents)
    twist_increments_b = jax.vmap(twist_increments)   # (B,N,4)×(B,N,4)→(B,N−1,4)

    def _proj_tangent(du, u):
        """Проекция шага на касательную к S³: du ← du − ⟨du,u⟩u (по-реперно)."""
        radial = jnp.sum(du * u, axis=-1, keepdims=True)
        return du - radial * u

    def _fix_endpoints(x, X0, XL):
        x = x.at[:, 0, :].set(X0)
        x = x.at[:, -1, :].set(XL)
        if freeze_w:
            x = x.at[..., 0].set(0.0)   # срез R³: w=Re=0 на всех узлах
        return x

    @jax.jit
    def run(key, x0, u0, X0, XL, U_A, U_B, T_schedule):
        """Прогон steps=len(T_schedule) ланжевен-шагов. Возврат:
        (x_final, u_final, rejected (B,)) — счётчик сингулярных отбраковок на ленту.
        step замыкает граничные данные (X0,XL,U_A,U_B) — константы прогона."""

        def step(carry, inp):
            x, u, rej = carry
            T, key = inp
            kx, ku = jax.random.split(key)
            # --- предложение ---
            gx = gx_b(x, u, U_A, U_B)
            gu = gu_b(x, u, U_A, U_B)
            nx = jax.random.normal(kx, x.shape) * jnp.sqrt(2.0 * lr * T)
            nu = jax.random.normal(ku, u.shape) * jnp.sqrt(2.0 * lr * T)
            x_new = _fix_endpoints(x - lr * gx + nx, X0, XL)
            du = _proj_tangent(-lr * gu + nu, u)
            u_new = normalize(u + du)
            # знаковая непрерывность: ближайший лифт к u(t)
            flip = jnp.sign(jnp.sum(u_new * u, axis=-1, keepdims=True) + 1e-30)
            u_new = u_new * flip
            # --- детекция сингулярностей (пер-лента, ADD §5) ---
            # (1) временной скачок лифта:
            overlap = jnp.sum(u_new * u, axis=-1)                  # (B,N)
            bad_u = jnp.any(overlap < k_sing, axis=-1)             # (B,)
            # (2) реверс касательной (бишоп-сингулярность):
            t_new = tan_b(x_new)                                   # (B,N,4)
            t_old = tan_b(x)
            tdot = jnp.sum(t_new[:, :-1] * t_new[:, 1:], axis=-1)   # (B,N−1)
            bad_t = jnp.any(tdot < k_tan, axis=-1)                 # (B,)
            # (3) ПРОСТРАНСТВЕННАЯ стена лифта: приращение твиста g_i.w меняет знак
            #     (связь проходит полуоборот) ⇒ «дырявый лифт», parity-опасно (ADD §5):
            g_old = twist_increments_b(u, t_old)                  # (B,N−1,4)
            g_new = twist_increments_b(u_new, t_new)
            wall = jnp.sign(g_new[..., 0]) != jnp.sign(g_old[..., 0])  # (B,N−1)
            bad_w = jnp.any(wall, axis=-1)                        # (B,)
            bad = (bad_u | bad_t | bad_w)[:, None, None]           # (B,1,1)
            # откат отклонённых лент к предыдущему состоянию
            x_out = jnp.where(bad, x, x_new)
            u_out = jnp.where(bad, u, u_new)
            rej = rej + bad[:, 0, 0].astype(rej.dtype)
            return (x_out, u_out, rej), None

        B = x0.shape[0]
        keys = jax.random.split(key, T_schedule.shape[0])
        rej0 = jnp.zeros((B,))
        (xf, uf, rej), _ = jax.lax.scan(step, (x0, u0, rej0), (T_schedule, keys))
        return xf, uf, rej

    # ------ SO(2)-ветка (ADD §8): состояние (x, φ), φ (B,N) ------
    def e_phi(x, phi, U_A, U_B):
        return e_total(x, u_of_phi(phi), U_A, U_B, params)

    gphi_b = jax.vmap(jax.grad(e_phi, argnums=1), in_axes=(0, 0, 0, 0))
    gxphi_b = jax.vmap(jax.grad(e_phi, argnums=0), in_axes=(0, 0, 0, 0))

    @jax.jit
    def run_so2(key, x0, phi0, X0, XL, U_A, U_B, T_schedule, W_target):
        """Прогон SO(2)-контроля. Возврат (x_final, phi_final, rejected (B,)).
        Отбраковка: t_i·t_{i+1}<−1+δ_tan ИЛИ |winding−W_target|≥wind_tol."""

        def step(carry, inp):
            x, phi, rej = carry
            T, key = inp
            kx, kp = jax.random.split(key)
            gx = gxphi_b(x, phi, U_A, U_B)
            gp = gphi_b(x, phi, U_A, U_B)
            nx = jax.random.normal(kx, x.shape) * jnp.sqrt(2.0 * lr * T)
            npi = jax.random.normal(kp, phi.shape) * jnp.sqrt(2.0 * lr * T)
            x_new = _fix_endpoints(x - lr * gx + nx, X0, XL)
            phi_new = phi - lr * gp + npi
            # --- отбраковка: тангенциальная + сохранение winding (ADD §8.6) ---
            t_new = tan_b(x_new)
            tdot = jnp.sum(t_new[:, :-1] * t_new[:, 1:], axis=-1)
            bad_t = jnp.any(tdot < k_tan, axis=-1)                 # (B,)
            bad_w = jnp.abs(winding(phi_new) - W_target) >= wind_tol  # (B,)
            bad = bad_t | bad_w                                    # (B,)
            x_out = jnp.where(bad[:, None, None], x, x_new)
            phi_out = jnp.where(bad[:, None], phi, phi_new)
            rej = rej + bad.astype(rej.dtype)
            return (x_out, phi_out, rej), None

        B = x0.shape[0]
        keys = jax.random.split(key, T_schedule.shape[0])
        rej0 = jnp.zeros((B,))
        (xf, pf, rej), _ = jax.lax.scan(step, (x0, phi0, rej0), (T_schedule, keys))
        return xf, pf, rej

    return {"run": run, "run_so2": run_so2, "lr": lr}
