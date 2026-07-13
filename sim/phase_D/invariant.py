"""Фаза D, стадия D0 — ℤ₂-инвариант оснащения (parity).

parity(config) ∈ {+1,−1}: произведение приращений лифта вдоль цепочки с
НЕПРЕРЫВНЫМ выбором знака, сверка с граничными аппаратными реперами (ADD §6).

Механизм: развёрнутый лифт L. Стартуем с u_0, выровненного по знаку к U_A. На каждой
связи берём внутреннее приращение g_i = u_i⁻¹⊗c_i⊗u_{i+1} (геометрическая связь c_i
вычтена, band4d.twist_increments), выбираем КОРОТКИЙ лифт (w≥0) — непрерывный выбор
знака — и домножаем L. В конце сравниваем L с U_B: parity = sign⟨L, U_B⟩.

π₁(SO(3))=ℤ₂: гладкая деформация не меняет parity; вставка 2π-поворота репера
переводит эндпоинт-лифт в антипод (parity=−1), 4π возвращает (+1). Сингулярный
шаг (скачок лифта через экватор) — это смена ветви; в динамике такие шаги
детектируются и отбраковываются (band4d.build_stepper), здесь — считаются отдельно.
"""

import jax
import jax.numpy as jnp
from jax import lax

from band4d import connection_quats, node_tangents
from ribbon_sim.frames import conj, normalize, quat_mul

_ID = jnp.array([1.0, 0.0, 0.0, 0.0])


def parity(x, u, U_A, U_B):
    """ℤ₂-parity одной конфигурации (ADD §6). x,u формы (N,4); U_A,U_B — (4,).

    Возврат скаляр ±1.0. Чистая функция (jit/vmap-совместима)."""
    t = node_tangents(x)
    c = connection_quats(t)                     # (N−1, 4)
    # приращения g_i = u_i⁻¹ ⊗ c_i ⊗ u_{i+1}
    g = quat_mul(quat_mul(conj(u[:-1]), c), u[1:])   # (N−1, 4)
    g_short = g * jnp.sign(g[:, 0:1] + 1e-30)   # непрерывный (короткий) лифт, w≥0

    L0 = normalize(u[0] * jnp.sign(jnp.sum(u[0] * U_A) + 1e-30))   # выравнивание к U_A

    def body(L, gi):
        return normalize(quat_mul(L, gi)), None

    L, _ = lax.scan(body, L0, g_short)
    return jnp.sign(jnp.sum(L * U_B) + 1e-30)


# --------------------------------------------------------------------------- #
#  Конструкторы конфигураций для unit-тестов (T-inv-1/2/3)
# --------------------------------------------------------------------------- #

def straight_rod(N, spacing=1.0):
    """Прямой стержень вдоль мнимой оси e1 (в срезе R³, w=0). x (N,4)."""
    x = jnp.zeros((N, 4))
    return x.at[:, 1].set(jnp.arange(N) * spacing)


def frame_rotation(N, total_angle, axis3=(0.0, 0.0, 1.0)):
    """Оснащение с равномерным поворотом репера на total_angle вдоль цепочки (ADD §6).

    u_i = exp(½·θ_i·axis) ∈ S³, θ_i = total_angle·i/(N−1). axis3 — ось в SO(3)
    (единичный 3-вектор). Даёт лифт-путь от (1,0,0,0) к (cos(θ/2), sin(θ/2)·axis).
    total_angle=2π ⇒ u_end=(−1,0,0,0) (антипод, parity=−1); 4π ⇒ (+1,0,0,0).
    """
    ax = jnp.asarray(axis3)
    ax = ax / jnp.linalg.norm(ax)
    theta = total_angle * jnp.arange(N) / (N - 1)          # (N,)
    half = 0.5 * theta
    w = jnp.cos(half)
    vec = jnp.sin(half)[:, None] * ax[None, :]             # (N,3)
    return jnp.concatenate([w[:, None], vec], axis=1)      # (N,4)


def parity_batch(x, u, U_A, U_B):
    """vmap parity по батчу (B,N,4)."""
    return jax.vmap(parity, in_axes=(0, 0, None, None))(x, u, U_A, U_B)
