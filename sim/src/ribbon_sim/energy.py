"""Энергия ленты (SPEC §2.2).

Фаза 1 — изотропная упругость: сумма квадратов геодезических расстояний
между соседними кадрами. Зажимы концов — квадратичные (±-симметричные).
"""

import jax.numpy as jnp

from .frames import axis, geodesic, quat_conj_mul, relative_log


def e_cosserat(q, k_b, k_t, spinor=False):
    """Анизотропная энергия Коссера через ЛОГ-отображение (atan2), R3-v1.

        E = Σ_i [ k_b·(ω_x² + ω_y²) + k_t·ω_z² ],   ω_i = log(q_i⁻¹⊗q_{i+1}) в фрейме i.

    ДЕФЕКТ (decisions.md, R3): многошаговый градиент (conj→mul→log_map/atan2) НЕ
    побитово ±-эквивариантен во float32 ⇒ нарушает ±-симметрию исходов. Оставлен для
    сверки старое/новое и мини-fp64-теста. Рабочий режим — cosserat_chordal.
    """
    omega = relative_log(q, spinor=spinor)  # (N-1, 3)
    bend = omega[..., 0] ** 2 + omega[..., 1] ** 2
    twist = omega[..., 2] ** 2
    return jnp.sum(k_b * bend + k_t * twist)


def e_cosserat_geo(q, k_b, k_t, spinor=False):
    """Анизотропная энергия Коссера на ГЕОДЕЗИЧЕСКОЙ жёсткости (R3-v3, архитектор):

        E_i = k_b·d²(q_i,q_{i+1}) + (k_t−k_b)·d²·z²/(x²+y²+z²+1e-12),

    d — геодезия текущего режима (geodesic|spinor), (w,x,y,z)=r_i=q_i⁻¹⊗q_{i+1}.
    z²/|vec|² = (доля оси поворота вдоль локального z)² ⇒ E = d²·(k_b·(1−f)+k_t·f),
    f=твист-доля. При k_t=k_b твист-член ТОЖДЕСТВЕННО 0 ⇒ E = geodesic (рабочая, R1).
    Жёсткая (растёт с углом, как d²) И почти симметричная (доминанта k_b·d² — geodesic,
    побитово эквивариантна; твист-член — несмещённый ULP-шум, как chordal).
    """
    d = geodesic(q[:-1], q[1:], spinor=spinor)            # (N-1,)
    r = quat_conj_mul(q[:-1], q[1:])                      # (N-1, 4)
    x, y, z = r[..., 1], r[..., 2], r[..., 3]
    twist_frac = z * z / (x * x + y * y + z * z + 1e-12)
    return jnp.sum(k_b * d * d + (k_t - k_b) * d * d * twist_frac)


def e_cosserat_chordal(q, k_b, k_t):
    """Хордальная энергия Коссера (R3-v2, решение архитектора): полином по компонентам
    относительного кватерниона r_i = q_i⁻¹⊗q_{i+1} = (w, x, y, z):

        E = Σ_i [ 4·k_b·(x² + y²) + 4·k_t·z² ].

    Без atan2/arccos/log_map ⇒ градиент ПОБИТОВО ±-эквивариантен (только произведения,
    квадраты и суммы). Для малых поворотов 4(x²+y²)≈ω_x²+ω_y², 4z²≈ω_z². Знаковая
    скрутка τ̃_i = 2·z_i (см. frames.total_twist).
    """
    r = quat_conj_mul(q[:-1], q[1:])  # (N-1, 4) = conj(q_i)⊗q_{i+1}
    x, y, z = r[..., 1], r[..., 2], r[..., 3]
    return jnp.sum(4.0 * k_b * (x * x + y * y) + 4.0 * k_t * (z * z))


def e_elastic(q, k_e, spinor=False, elastic="geodesic", k_b=1.0, k_t=1.0):
    """E_elastic = k_e * Σ_{i=0}^{N-2} L(q[i], q[i+1])   (SPEC §2.2).

    q формы (N, 4). Изотропная фаза 1: все соседние связи равноправны.
    Метрика связи (elastic):
      - "geodesic": L = d² = arccos(|<p,q>|)²  — по SPEC §2.2 (по умолчанию);
      - "spinor":   L = arccos(<p,q>)²         — БЕЗ модуля, различает q и −q
        (720°-периодичность, SPEC §2.2 фаза 3, R5). d(q,−q)=π ⇒ полуоборотные
        «кинки» между соседями энергетически штрафуются и застревают при T=0;
      - "chordal":  L = 1 − <p,q>²             — гладкая альтернатива без arccos,
        проверка робастности (фаза C). ±-симметрична (квадрат снимает знак).
    """
    if elastic == "cosserat_geo":
        return e_cosserat_geo(q, k_b, k_t, spinor=spinor)
    if elastic == "cosserat_chordal":
        return e_cosserat_chordal(q, k_b, k_t)
    if elastic == "cosserat":
        return e_cosserat(q, k_b, k_t, spinor=spinor)
    if elastic == "chordal":
        c = jnp.sum(q[:-1] * q[1:], axis=-1)  # (N-1,)
        return k_e * jnp.sum(1.0 - c * c)
    if elastic == "spinor":
        spinor = True  # геодезия без модуля
    elif elastic != "geodesic":
        raise ValueError(f"неизвестный режим elastic: {elastic!r}")
    d = geodesic(q[:-1], q[1:], spinor=spinor)  # (N-1,)
    return k_e * jnp.sum(d * d)


def e_clamp(q_end, axis_vec, k_c):
    """E_clamp = −k_c * (n·axis)²   (SPEC §2.2).

    Квадратичный зажим: два минимума при n = ±axis, знак не предпочтён.
    """
    n = axis(q_end)  # (3,)
    proj = jnp.dot(n, axis_vec)
    return -k_c * proj * proj


def e_total(q, a, b, k_e, k_c, spinor=False, elastic="geodesic", k_b=1.0, k_t=1.0):
    """Полная энергия одной ленты (SPEC §2.2):

        E_total = E_elastic + E_clamp_A + E_clamp_B

    q формы (N, 4); a, b — единичные 3-векторы осей зажимов; скаляр на выходе.
    k_b, k_t — жёсткости изгиба/скрутки (только для elastic="cosserat", R3).
    Функция чистая от СЫРОГО q (без внутренней нормировки) — так конечные
    разности и jax.grad видят одну и ту же функцию (тест SPEC §7).
    """
    return (
        e_elastic(q, k_e, spinor=spinor, elastic=elastic, k_b=k_b, k_t=k_t)
        + e_clamp(q[0], a, k_c)
        + e_clamp(q[-1], b, k_c)
    )
