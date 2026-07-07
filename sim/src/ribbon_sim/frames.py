"""Кватернионы и геометрия на S³ (SPEC §2.1).

Соглашение: кватернион скаляр-первый q = (w, x, y, z), q ∈ S³ ⊂ R⁴.
Ось конца n(q) = R(q)·e_z — третий столбец матрицы поворота.
"""

import jax.numpy as jnp

# Порог обрезки аргумента arccos. В float32 машинный eps ≈ 1.2e-7, поэтому
# 1 − 1e-7 неотличимо от 1.0; берём 1e-6, чтобы arccos и его градиент
# (∝ 1/sqrt(1−c²)) оставались конечными у c = ±1 (совпадающие/антиподные кадры).
GEODESIC_EPS = 1e-6


def normalize(q):
    """Проекция обратно на S³ (SPEC §2.3): q / |q| вдоль последней оси."""
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm


def rotmat(q):
    """Матрица поворота 3×3 из единичного кватерниона (w, x, y, z).

    Работает по батчам: q формы (..., 4) → (..., 3, 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    row0 = jnp.stack([r00, r01, r02], axis=-1)
    row1 = jnp.stack([r10, r11, r12], axis=-1)
    row2 = jnp.stack([r20, r21, r22], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def axis(q):
    """Ось конца n(q) = R(q)·e_z — третий столбец R(q) (SPEC §2.1).

    q формы (..., 4) → n формы (..., 3). Дешевле, чем строить всю rotmat.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    nx = 2 * (x * z + w * y)
    ny = 2 * (y * z - w * x)
    nz = 1 - 2 * (x * x + y * y)
    return jnp.stack([nx, ny, nz], axis=-1)


def quat_mul(q1, q2):
    """Произведение Гамильтона (скаляр-первый). Композиция поворотов:
    R(quat_mul(q1, q2)) = R(q1) · R(q2). Работает по батчам (..., 4)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.stack([w, x, y, z], axis=-1)


def log_map(q):
    """Логарифм единичного кватерниона → вектор поворота (axis·angle) ∈ R³.

    angle = 2·atan2(|v|, w). Понадобится в R3 (стержень Коссера, SPEC §3):
    разложение относительного поворота на изгиб/скрутку.
    """
    w = q[..., 0]
    v = q[..., 1:]
    vn = jnp.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * jnp.arctan2(vn[..., 0], w)  # (...,)
    safe_vn = jnp.where(vn > 1e-12, vn, 1.0)
    return (angle[..., None] / safe_vn) * v


def conj(q):
    """Сопряжение единичного кватерниона = обратный: (w, −x, −y, −z)."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_conj_mul(p, q):
    """Относительный поворот r = p⁻¹⊗q = conj(p)⊗q (в материальном фрейме p)."""
    return quat_mul(conj(p), q)


def exp_quat(omega):
    """Экспонента so(3)→S³: вектор поворота omega ∈ R³ → единичный кватернион.

    omega формы (..., 3) → (..., 4). q = (cos|ω|/2, sin(|ω|/2)·ω/|ω|).
    """
    theta = jnp.linalg.norm(omega, axis=-1, keepdims=True)  # (...,1)
    half = 0.5 * theta
    # sin(half)/theta с пределом 1/2 при theta→0
    safe = jnp.where(theta > 1e-12, theta, 1.0)
    coef = jnp.where(theta > 1e-12, jnp.sin(half) / safe, 0.5)
    w = jnp.cos(half)
    return jnp.concatenate([w, coef * omega], axis=-1)


def total_twist(q):
    """Суммарная скрутка Tw = Σ_i τ̃_i, где τ̃_i = 2·z_i — хордальная скрутка
    (z-компонента относительного кватерниона r_i = q_i⁻¹⊗q_{i+1}), R4/архитектор.

    q формы (..., N, 4) → (...,). Без atan2/log_map: гладко и знаково-непрерывно;
    z различает Tw=0 и Tw=2π-секторы (спинорная ℤ₂-структура).
    """
    r = quat_conj_mul(q[..., :-1, :], q[..., 1:, :])      # (..., N-1, 4)
    return jnp.sum(2.0 * r[..., 3], axis=-1)


def twist_free_init(key, shape, total_twist=0.0, sigma_bend=0.25):
    """Случайная лента в секторе заданной суммарной скрутки Tw=Σ2z_i (R4).

    Каждая связь: относительный кватернион r_i = (w, bx, by, z_link) с фиксированной
    z-компонентой z_link = Tw/(2(N−1)) (⇒ Σ2z_i = Tw ТОЧНО) и случайным изгибом
    (bx,by) ~ Normal(0, sigma_bend). Мера объявлена заранее (note §2.7). shape=(B,N)→(B,N,4).
    """
    import jax.random as jr

    B, N = shape
    k0, kb = jr.split(key)
    q0 = haar_quaternions(k0, (B,))                       # (B, 4)
    z_link = total_twist / (2.0 * (N - 1))
    bend = jr.normal(kb, (B, N - 1, 2)) * sigma_bend
    x, y = bend[..., 0], bend[..., 1]
    z = jnp.full_like(x, z_link)
    # Ограничиваем изгиб, чтобы x²+y²+z² ≤ 0.95 (w²≥0.05): z (скрутка) остаётся ТОЧНОЙ,
    # r всегда единичный, без пола 1e-6, искажавшего z у хвостовых лент.
    b2 = x * x + y * y
    max_b2 = jnp.maximum(0.95 - z * z, 1e-6)
    scale = jnp.where(b2 > max_b2, jnp.sqrt(max_b2 / jnp.maximum(b2, 1e-12)), 1.0)
    x, y = x * scale, y * scale
    w = jnp.sqrt(1.0 - x * x - y * y - z * z)
    r = jnp.stack([w, x, y, z], axis=-1)                  # (B, N-1, 4), уже единичный
    q = q0
    frames = [q0]
    for i in range(N - 1):
        q = normalize(quat_mul(q, r[:, i, :]))            # нормировка на шаге: без дрейфа |q|
        frames.append(q)
    return jnp.stack(frames, axis=1)                      # (B, N, 4)


def relative_log(q, spinor=False):
    """Лог-отображение относительных поворотов соседей в МАТЕРИАЛЬНОМ фрейме (R3).

    r_i = q_i⁻¹ ⊗ q_{i+1} — поворот из фрейма i в i+1 в координатах i.
    ω_i = log_map(r_i) ∈ R³: ω_z — скрутка, (ω_x, ω_y) — изгиб (дискретный Коссера).

    q формы (..., N, 4) → ω формы (..., N−1, 3). При spinor=False берём короткий
    поворот (r → знак w≥0, |ω|<π); spinor=True сохраняет ветвь (для твиста ±2π, R4).
    """
    r = quat_mul(conj(q[..., :-1, :]), q[..., 1:, :])  # (..., N-1, 4)
    if not spinor:
        r = r * jnp.sign(r[..., 0:1] + 1e-30)  # короткий поворот: w ≥ 0
    return log_map(r)


def geodesic(p, q, spinor=False):
    """Геодезическое расстояние на S³ (SPEC §2.2).

    Фаза 1 (spinor=False): d(p, q) = arccos(|<p, q>|) — нить, не различает q/−q.
    Спинорный режим (spinor=True): d(p, q) = arccos(<p, q>) — различает q и −q
    (720°-периодичность, SPEC §2.2, фаза 3).

    Аргумент arccos обрезается на GEODESIC_EPS от ±1 ради конечного градиента.
    """
    c = jnp.sum(p * q, axis=-1)
    if not spinor:
        c = jnp.abs(c)
    c = jnp.clip(c, -1.0 + GEODESIC_EPS, 1.0 - GEODESIC_EPS)
    return jnp.arccos(c)


def haar_quaternions(key, shape):
    """Haar-равномерная выборка кватернионов (SPEC §2.4).

    shape — форма БЕЗ последней оси-4, напр. (B, N). Возврат (B, N, 4).
    Нормированный гауссов вектор в R⁴ равномерен на S³ (мера Хаара).
    """
    import jax.random as jr

    g = jr.normal(key, shape + (4,))
    return normalize(g)
