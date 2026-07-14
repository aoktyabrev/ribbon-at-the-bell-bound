"""Фаза D, стадия D1 — протокол измерения (см. D1_prereg.md, D1_prereg_addendum.md).

Осевые квадратичные зажимы концов −k_c(n_end·a)², n_end — осевой вектор SO(3)-репера
конца в НОРМАЛЬНЫХ координатах (заведомо SO(3)-факторизуем, слеп к лифту ℤ₂ —
эталон слепоты AX). Обе ветви (±) доступны по построению (два минимума n=±a).

Считывающая ось (нормальные координаты): ê = REF_AXIS = (0,0,1). Осевой вектор конца
n_end = rotmat(u_end)·ê ∈ R³. Аппаратные оси a,b ∈ R³ (в нормальных коорд среза):
a=(0,0,1); θ=0 ⇒ b=a; θ=π ⇒ b=−a. (В D1 разрешены только θ∈{0,π}.)

Секторы: чётный — u≡const (φ≡0); нечётный — 2π-накрутка (u_of_phi, φ:0→2π).
Исход — СЫРЫЕ знаки (s,t)=(sign(n_A·a), sign(n_B·b)); DEGENERATE — если бассейн
не определился. Зеркальные пары (a,b)↔(−a,−b) + зеркало оснащения u→u⊗j.

Всё чистый JAX, CPU float64. Идентификаторы английские; формулы — ADD (D1) §k.
"""

import jax
import jax.numpy as jnp
from jax import lax

import band4d as B
from ribbon_sim.frames import normalize, quat_mul, rotmat

REF_AXIS = jnp.array([0.0, 0.0, 1.0])   # ê в нормальных координатах (ADD §1)
J_QUAT = jnp.array([0.0, 1.0, 0.0, 0.0])  # зеркало оснащения u→u⊗j


# --------------------------------------------------------------------------- #
#  Осевой вектор конца и осевой зажим
# --------------------------------------------------------------------------- #

def end_axis(u_end):
    """Осевой вектор SO(3)-репера конца n_end = rotmat(u_end)·ê ∈ R³ (норм. коорд).
    Зависит ТОЛЬКО от rotmat(u_end) ⇒ слеп к лифту u vs −u (SO(3)-факторизуем)."""
    return rotmat(u_end) @ REF_AXIS


def e_axial_clamp(u_end, a, k_c):
    """Осевой квадратичный зажим −k_c·(n_end·a)² (ADD §2). Два минимума n_end=±a."""
    proj = jnp.dot(end_axis(u_end), a)
    return -k_c * proj * proj


def e_meas(x, u, a, b, params):
    """Полная энергия измерения (одна кривая): упругость D0 + ОСЕВЫЕ зажимы концов.
    params — dict k_s,k_b,k_f,k_c,ell. Чистая от сырых (x,u)."""
    t = B.node_tangents(x)
    return (
        B.e_stretch(x, params["k_s"], params["ell"])
        + B.e_bend(t, params["k_b"])
        + B.e_twist(u, t, params["k_f"])
        + e_axial_clamp(u[0], a, params["k_c"])
        + e_axial_clamp(u[-1], b, params["k_c"])
    )


# --------------------------------------------------------------------------- #
#  Классификация исхода (сырые s,t) + DEGENERATE
# --------------------------------------------------------------------------- #

# Порог «определённости» бассейна: |n_end·a| ниже — ось не выбрала знак (DEGENERATE).
DECISION_EPS = 0.2


def classify_raw(u, a, b):
    """Сырые знаки (s,t)=(sign(n_A·a), sign(n_B·b)) и |проекции| (для DEGENERATE).
    u (N,4) → (s, t, |projA|, |projB|). sign(0)→+1."""
    pa = jnp.dot(end_axis(u[0]), a)
    pb = jnp.dot(end_axis(u[-1]), b)
    s = jnp.where(pa >= 0, 1, -1)
    t = jnp.where(pb >= 0, 1, -1)
    return s, t, jnp.abs(pa), jnp.abs(pb)


def classify_batch(u, a, b):
    """(B,N,4) → (s (B,), t (B,), degenerate (B,) bool)."""
    s, t, pa, pb = jax.vmap(classify_raw, in_axes=(0, None, None))(u, a, b)
    degen = (pa < DECISION_EPS) | (pb < DECISION_EPS)
    return s, t, degen


# --------------------------------------------------------------------------- #
#  Аппаратные оси и приготовление секторов
# --------------------------------------------------------------------------- #

def apparatus_axes(theta_is_pi):
    """(a, b): a=(0,0,1); b=a при θ=0, b=−a при θ=π. Возврат 3-векторы."""
    a = REF_AXIS
    b = jnp.where(theta_is_pi, -REF_AXIS, REF_AXIS)
    return a, b


def frame_for_axis(sign):
    """Репер конца u с осевым вектором n=sign·ê: identity (s=+) или 180° вокруг x̂ (s=−).
    rotmat(identity)·ê=ê; rotmat((0,1,0,0))·ê=−ê."""
    return jnp.where(sign > 0,
                     jnp.array([1.0, 0.0, 0.0, 0.0]),
                     jnp.array([0.0, 1.0, 0.0, 0.0]))


def branch_frame_signs(s_end, t_end, theta_pi):
    """Знаки граничных реперов, дающие наблюдаемые (s,t)=(sign(n_A·a),sign(n_B·b)).
    n_A=s·ê·a→sign_A=s; n_B·b с b=±ê ⇒ sign_B = t·sign(b·ê) = t (θ=0) / −t (θ=π)."""
    sign_b = jnp.where(theta_pi, -1, 1)
    return s_end, t_end * sign_b


def prep_sector(N, sector_odd, s_end, t_end, theta_pi, batch, key,
                sigma_pos=0.05, sigma_frame=0.0):
    """Приготовление ячейки: сектор (odd/even) с НАБЛЮДАЕМЫМИ ветвями (s_end,t_end)
    при данном θ (b=a при θ=0, b=−a при θ=π).

    Позиции — прямой стержень в срезе R³ (+шум формы). Оснащение: φ-профиль
    интерполирует реперы концов; в НЕЧЁТНОМ секторе +2π-накрутка (parity=−1), в
    чётном 0. Граничные реперы дают n_A=s·a, n_B=t·b. Возврат dict x0,u0,X0,XL.
    """
    import jax.random as jr
    sr = _straight_rod(N)
    kx, kf = jr.split(key)
    sa, sb = branch_frame_signs(s_end, t_end, theta_pi)
    base_tw = jnp.where(sector_odd, 2.0 * jnp.pi, 0.0)
    phi = base_tw * jnp.arange(N) / (N - 1)                    # (N,)
    u = B.u_of_phi(phi)                                        # (N,4)
    uA = quat_mul(frame_for_axis(sa), u[0])
    uB = quat_mul(frame_for_axis(sb), u[-1])
    u = u.at[0].set(uA).at[-1].set(uB)
    # батч + шум формы (позиции в срезе) и опц. реперов
    x0 = jnp.tile(sr, (batch, 1, 1))
    if sigma_pos > 0:
        noise = sigma_pos * jr.normal(kx, (batch, N, 4))
        noise = noise.at[..., 0].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
        x0 = x0 + noise
    u0 = jnp.tile(u, (batch, 1, 1))
    if sigma_frame > 0:
        du = sigma_frame * jr.normal(kf, (batch, N, 4))
        u0 = normalize(u0 + du)
    X0 = jnp.tile(sr[0], (batch, 1))
    XL = jnp.tile(sr[-1], (batch, 1))
    return dict(x0=x0, u0=u0, X0=X0, XL=XL)


def prep_dynamics(N, sector_odd, batch, key, sigma_pos=0.06, tilt=None):
    """НЕсмещённое приготовление для D1-C: сектор задан (odd/even) интерьерной
    накруткой; граничные реперы наклонены так, что осевой вектор n_end РАВНОМЕРЕН на
    S² (полярный угол arccos(1−2ξ)) ⇒ ветвь НЕ выбрана заранее, бассейн решает
    симметрично. Позиции — стержень в срезе + шум формы. Возврат dict x0,u0,X0,XL."""
    import jax.random as jr
    sr = _straight_rod(N)
    kx, ka, kb, kaxa, kaxb = jr.split(key, 5)
    base_tw = jnp.where(sector_odd, 2.0 * jnp.pi, 0.0)
    phi = base_tw * jnp.arange(N) / (N - 1)
    u = B.u_of_phi(phi)                                        # (N,4)
    x0 = jnp.tile(sr, (batch, 1, 1))
    noise = sigma_pos * jr.normal(kx, (batch, N, 4))
    noise = noise.at[..., 0].set(0.0).at[:, 0].set(0.0).at[:, -1].set(0.0)
    x0 = x0 + noise
    u0 = jnp.tile(u, (batch, 1, 1))
    # наклон граничных реперов на tilt вокруг случайной оси (в каждой реплике)
    def tilt_quat(k, n):
        axis = normalize(jr.normal(k, (n, 3)))
        # полярный угол для РАВНОМЕРНОГО n_end на S²: cos α = 1−2ξ ⇒ α=arccos(...)
        ang = jnp.arccos(jnp.clip(1.0 - 2.0 * jr.uniform(jr.fold_in(k, 1), (n,)), -1.0, 1.0))
        half = 0.5 * ang
        return jnp.concatenate([jnp.cos(half)[:, None],
                                jnp.sin(half)[:, None] * axis], axis=1)
    qA = tilt_quat(ka, batch); qB = tilt_quat(kb, batch)
    u0 = u0.at[:, 0].set(normalize(quat_mul(qA, u0[:, 0])))
    u0 = u0.at[:, -1].set(normalize(quat_mul(qB, u0[:, -1])))
    X0 = jnp.tile(sr[0], (batch, 1))
    XL = jnp.tile(sr[-1], (batch, 1))
    return dict(x0=x0, u0=u0, X0=X0, XL=XL)


def _straight_rod(N, spacing=1.0):
    """Прямой стержень вдоль мнимой оси e1 (в срезе R³, w=0). x (N,4)."""
    x = jnp.zeros((N, 4))
    return x.at[:, 1].set(jnp.arange(N) * spacing)


def mirror_frames(u):
    """Зеркало оснащения u→u⊗j (ADD §7 D0)."""
    return quat_mul(u, J_QUAT)


# --------------------------------------------------------------------------- #
#  Минимизатор (T=0) с осевыми зажимами и детекцией сингулярностей
# --------------------------------------------------------------------------- #

def build_minimizer(params, lr=5e-3, freeze_w=False,
                    delta_tan=B.DELTA_TAN, delta_sing=B.DELTA_SING):
    """T=0 (или T>0) релаксация e_meas на (x,u). Детекция сингулярностей как в D0:
    временной скачок лифта, реверс касательной, ПРОСТРАНСТВЕННАЯ стена g_i.w=0.
    Возврат dict run(key,x0,u0,a,b,T_sched)→(x,u,rejected,parity_changes)."""
    k_sing = 1.0 - float(delta_sing)
    k_tan = -1.0 + float(delta_tan)

    def e1(x, u, a, b):
        return e_meas(x, u, a, b, params)

    gx_b = jax.vmap(jax.grad(e1, 0), in_axes=(0, 0, None, None))
    gu_b = jax.vmap(jax.grad(e1, 1), in_axes=(0, 0, None, None))
    tan_b = jax.vmap(B.node_tangents)
    twinc_b = jax.vmap(B.twist_increments)

    def _fix_ends(x, X0, XL):
        x = x.at[:, 0, :].set(X0).at[:, -1, :].set(XL)
        if freeze_w:
            x = x.at[..., 0].set(0.0)
        return x

    @jax.jit
    def run(key, x0, u0, X0, XL, a, b, T_schedule):
        def step(carry, inp):
            x, u, rej = carry
            T, k = inp
            kx, ku = jax.random.split(k)
            gx = gx_b(x, u, a, b)
            gu = gu_b(x, u, a, b)
            nx = jax.random.normal(kx, x.shape) * jnp.sqrt(2.0 * lr * T)
            nu = jax.random.normal(ku, u.shape) * jnp.sqrt(2.0 * lr * T)
            x_new = _fix_ends(x - lr * gx + nx, X0, XL)
            du = -lr * gu + nu
            du = du - jnp.sum(du * u, axis=-1, keepdims=True) * u   # касательный
            u_new = normalize(u + du)
            flip = jnp.sign(jnp.sum(u_new * u, axis=-1, keepdims=True) + 1e-30)
            u_new = u_new * flip
            # сингулярности
            overlap = jnp.sum(u_new * u, axis=-1)
            bad_u = jnp.any(overlap < k_sing, axis=-1)
            t_old = tan_b(x); t_new = tan_b(x_new)
            tdot = jnp.sum(t_new[:, :-1] * t_new[:, 1:], axis=-1)
            bad_t = jnp.any(tdot < k_tan, axis=-1)
            g_old = twinc_b(u, t_old); g_new = twinc_b(u_new, t_new)
            wall = jnp.sign(g_new[..., 0]) != jnp.sign(g_old[..., 0])
            bad_w = jnp.any(wall, axis=-1)
            bad = (bad_u | bad_t | bad_w)
            x_out = jnp.where(bad[:, None, None], x, x_new)
            u_out = jnp.where(bad[:, None, None], u, u_new)
            rej = rej + bad.astype(rej.dtype)
            return (x_out, u_out, rej), None

        keys = jax.random.split(key, T_schedule.shape[0])
        rej0 = jnp.zeros((x0.shape[0],))
        (xf, uf, rej), _ = lax.scan(step, (x0, u0, rej0), (T_schedule, keys))
        return xf, uf, rej

    return {"run": run, "lr": lr}
