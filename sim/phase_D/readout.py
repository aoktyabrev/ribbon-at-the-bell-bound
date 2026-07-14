"""D1-B. Кандидаты считывания + тесты слепоты/локальности (D1_prereg §D-H3c, аддендум §5).

Каждый кандидат даёт ПАРУ знаков (левый, правый) ∈ {±1}² — аналог (s,t) для AX.
- AX  : sign(n_end·a) — осевой, SO(3)-факторизуемый эталон слепоты.
- A   : sign(w-компоненты фактической считывающей нормали в R⁴) — геометрия+SO(3).
- C   : sign(w) ближайшего интерьерного узла — чистая геометрия среза.
- Bloc: parity развёрнутого лифта в окне k узлов относительно аппаратного репера.

Тест слепоты: пары (u, u') с ТОЖДЕСТВЕННЫМ осевым полем n_i, но ПРОТИВОПОЛОЖНЫМ
классом лифта (u'_i = R(ψ_i,n_i)⊗u_i, ψ:0→2π — вращение вокруг локальной оси n_i,
не меняющее n_i). Совпадение кандидата на всех парах = SO(3)-факторизуемость.
Тест локальности (Bloc): стабилизация по k∈{2,4,8}; иначе BAKED-IN.
"""

import jax
import jax.numpy as jnp
import numpy as np

import band4d as B
import measurement as M
from ribbon_sim.frames import conj, normalize, quat_mul, rotmat

E0 = jnp.array([1.0, 0.0, 0.0, 0.0])   # ось w (4-е измерение) = скалярная компонента


# --------------------------------------------------------------------------- #
#  Кандидаты (одна кривая) → (left, right) ∈ {±1}
# --------------------------------------------------------------------------- #

def cand_AX(x, u, a, b):
    """(AX) осевой SO(3): (sign(n_A·a), sign(n_B·b))."""
    nA = M.end_axis(u[0]); nB = M.end_axis(u[-1])
    return _sgn(jnp.dot(nA, a)), _sgn(jnp.dot(nB, b))


def cand_A(x, u, a, b):
    """(A) знак w-компоненты фактической считывающей нормали в R⁴ у каждого конца.
    N_read = P(t_end)ᵀ(rotmat(u_end)·ê) — 4-вектор; берём его w=index0 (выпучивание)."""
    t = B.node_tangents(x)
    nrmA = _readout_normal_4d(u[0], t[0])
    nrmB = _readout_normal_4d(u[-1], t[-1])
    return _sgn(nrmA[0]), _sgn(nrmB[0])


def cand_C(x, u, a, b):
    """(C) sign(w) ближайшего интерьерного узла: x_1.w (лево), x_{N−2}.w (право)."""
    return _sgn(x[1][0]), _sgn(x[-2][0])


def cand_Bloc(x, u, a, b, k):
    """(Bloc) parity развёрнутого лифта окна k у каждого конца относительно
    аппаратного репера. Продукт sign(g_i.w) первых/последних k связей × граничный знак."""
    t = B.node_tangents(x)
    g = B.twist_increments(u, t)          # (N−1,4)
    # левое окно: первые k связей
    left = jnp.prod(jnp.sign(g[:k, 0] + 1e-30))
    # правое окно: последние k связей
    right = jnp.prod(jnp.sign(g[-k:, 0] + 1e-30))
    return _sgn(left), _sgn(right)


def _readout_normal_4d(u_end, t_end):
    """Фактическая считывающая нормаль в R⁴: N = Σ_l (rotmat(u)·ê)[l] · P(t)[l]."""
    coords = rotmat(u_end) @ M.REF_AXIS          # (3,) норм. коорд
    P = B.canonical_normal_basis(t_end)          # (3,4)
    return coords @ P                            # (4,)


def _sgn(v):
    return jnp.where(v >= 0, 1, -1)


CANDIDATES = {
    "AX": cand_AX,
    "A": cand_A,
    "C": cand_C,
    "Bloc2": lambda x, u, a, b: cand_Bloc(x, u, a, b, 2),
    "Bloc4": lambda x, u, a, b: cand_Bloc(x, u, a, b, 4),
    "Bloc8": lambda x, u, a, b: cand_Bloc(x, u, a, b, 8),
}


def eval_all(x, u, a, b):
    """Все кандидаты на одной кривой → dict name→(left,right)."""
    return {name: fn(x, u, a, b) for name, fn in CANDIDATES.items()}


def eval_all_batch(xb, ub, a, b):
    """Батч → dict name→(L (B,), R (B,)) numpy."""
    out = {}
    for name, fn in CANDIDATES.items():
        L, R = jax.vmap(fn, in_axes=(0, 0, None, None))(xb, ub, a, b)
        out[name] = (np.asarray(L), np.asarray(R))
    return out


# --------------------------------------------------------------------------- #
#  Тест слепоты
# --------------------------------------------------------------------------- #

def lift_twin(u):
    """u' с ТОЖДЕСТВЕННЫМ осевым полем n_i, но ПРОТИВОПОЛОЖНЫМ классом лифта.
    u'_i = R(ψ_i, n_i) ⊗ u_i, ψ_i=2π·i/(N−1); n_i=rotmat(u_i)·ê. Вращение вокруг
    локальной оси n_i не меняет n_i, но накручивает лифт на 2π (flip ℤ₂). u (N,4)→(N,4)."""
    N = u.shape[0]
    psi = 2.0 * jnp.pi * jnp.arange(N) / (N - 1)          # (N,)
    n = jax.vmap(lambda q: rotmat(q) @ M.REF_AXIS)(u)     # (N,3) осевые векторы
    half = 0.5 * psi
    rot = jnp.concatenate([jnp.cos(half)[:, None],
                           jnp.sin(half)[:, None] * n], axis=1)   # (N,4) кватернион R(ψ,n)
    return normalize(quat_mul(rot, u))


def blindness_test(xb, ub, a, b):
    """Для каждого кандидата: доля пар (u,u'), где считывание СОВПАДАЕТ.
    Совпадение на всех парах = SO(3)-факторизуемость (слепота к лифту).
    Проверка: осевое поле n_i у пар тождественно (макс|Δn|)."""
    ub2 = jax.vmap(lift_twin)(ub)
    # контроль: осевые поля тождественны
    n1 = jax.vmap(lambda u: jax.vmap(lambda q: rotmat(q) @ M.REF_AXIS)(u))(ub)
    n2 = jax.vmap(lambda u: jax.vmap(lambda q: rotmat(q) @ M.REF_AXIS)(u))(ub2)
    max_dn = float(jnp.max(jnp.abs(n1 - n2)))
    r1 = eval_all_batch(xb, ub, a, b)
    r2 = eval_all_batch(xb, ub2, a, b)
    out = {}
    for name in CANDIDATES:
        agreeL = (r1[name][0] == r2[name][0])
        agreeR = (r1[name][1] == r2[name][1])
        frac = float((agreeL & agreeR).mean())
        out[name] = frac        # 1.0 = слеп (SO(3)-факторизуем), <1 = видит лифт
    return out, max_dn


# --------------------------------------------------------------------------- #
#  Тест локальности (Bloc по k)
# --------------------------------------------------------------------------- #

def locality_test(xb, ub, a, b, ks=(2, 4, 8)):
    """Bloc(k) по k: доля конфигураций, где Bloc(k) совпадает с Bloc(max k).
    Стабилизация при k≤8 ⇒ локален; иначе (стабилизация лишь при k~N/2) — BAKED-IN."""
    ref_k = max(ks)
    Lref, Rref = jax.vmap(lambda x, u: cand_Bloc(x, u, a, b, ref_k),
                          in_axes=(0, 0))(xb, ub)
    Lref, Rref = np.asarray(Lref), np.asarray(Rref)
    out = {}
    for k in ks:
        Lk, Rk = jax.vmap(lambda x, u: cand_Bloc(x, u, a, b, k), in_axes=(0, 0))(xb, ub)
        out[k] = float(((np.asarray(Lk) == Lref) & (np.asarray(Rk) == Rref)).mean())
    return out
