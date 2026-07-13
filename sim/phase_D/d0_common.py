"""Общая инфраструктура прогонов V1–V3 фазы D0 (см. D0_prereg*.md).

Приготовления, наблюдаемые, блочная релаксация, KS-тест, зеркальные пары.
Все прогоны: CPU-JAX, float64. Идентификаторы английские, формулы — ADD §k.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import stats

import band4d as B
import invariant as I
from ribbon_sim.frames import quat_mul

ID = jnp.array([1.0, 0.0, 0.0, 0.0])
J_QUAT = jnp.array([0.0, 1.0, 0.0, 0.0])   # зеркало оснащения u→u⊗j (ADD §7)


# --------------------------------------------------------------------------- #
#  Приготовления
# --------------------------------------------------------------------------- #

def phi_localized(N, total, center=0.5, width=0.4):
    """Профиль φ(i): гладкая ступень 0→total, локализованная в середине (ADD §8.2).
    Носитель твиста — «солитон» шириной ~width·L в центре. width подобрана так,
    чтобы max Δφ/связь < π (иначе связь-полуоборот даёт артефактный parity: max
    Δφ ≈ total·3/(2·width·(N−1)); для 4π, N≥48, width≥0.4 ⇒ < π)."""
    s = jnp.arange(N) / (N - 1)
    theta = 0.5 * (1.0 + jnp.tanh((s - center) / (width / 3.0)))
    theta = theta - theta[0]
    return theta * (total / theta[-1])   # нормировка: φ_{N−1}−φ_0 = total ТОЧНО ⇒ W точен


def prep_localized(N, total, batch, width=0.4, key=None):
    """Приготовление локализованного твиста total (для V1). Возврат dict:
    x0 (B,N,4) — прямой стержень в срезе R³; u0 (B,N,4)=u_of_phi; phi0 (B,N);
    X0,XL (B,4) — фикс концов; UA,UB (B,4)=ID (аппаратные реперы)."""
    sr = I.straight_rod(N)
    phi1 = phi_localized(N, total, width=width)
    x0 = jnp.tile(sr, (batch, 1, 1))
    phi0 = jnp.tile(phi1, (batch, 1))
    u0 = B.u_of_phi(phi0)
    X0 = jnp.tile(sr[0], (batch, 1))
    XL = jnp.tile(sr[-1], (batch, 1))
    UA = jnp.tile(ID, (batch, 1))
    UB = jnp.tile(ID, (batch, 1))
    return dict(x0=x0, u0=u0, phi0=phi0, X0=X0, XL=XL, UA=UA, UB=UB)


def prep_uniform(N, total, batch, sigma=0.0, key=None):
    """Приготовление РАВНОМЕРНОГО твиста total (для V2/V3 и Tw=0-эталона).
    sigma>0 добавляет разнообразие форм позиций (шум в срезе, зеркальные пары честнее)."""
    sr = I.straight_rod(N)
    theta = total * jnp.arange(N) / (N - 1)
    u1 = B.u_of_phi(theta)
    x0 = jnp.tile(sr, (batch, 1, 1))
    if sigma > 0 and key is not None:
        noise = sigma * jr.normal(key, (batch, N, 4))
        noise = noise.at[..., 0].set(0.0)             # держим в срезе
        x0 = x0 + noise.at[:, 0].set(0.0).at[:, -1].set(0.0)
    u0 = jnp.tile(u1, (batch, 1, 1))
    phi0 = jnp.tile(theta, (batch, 1))
    X0 = jnp.tile(sr[0], (batch, 1))
    XL = jnp.tile(sr[-1], (batch, 1))
    UA = jnp.tile(ID, (batch, 1))
    UB = jnp.tile(ID, (batch, 1))
    return dict(x0=x0, u0=u0, phi0=phi0, X0=X0, XL=XL, UA=UA, UB=UB)


def mirror_frames(u):
    """Зеркальная пара оснащения u→u⊗j (ADD §7): ±-контроль (перенос практики A–C)."""
    return quat_mul(u, J_QUAT)


def mirror_prep(prep):
    """Зеркальный prep: оснащение И реперы-ссылки u→u⊗j (ИЗОМЕТРИЯ — энергия/parity
    инвариантны). ±-контроль дискретизации: динамика и parity должны совпасть."""
    m = dict(prep)
    m["u0"] = mirror_frames(prep["u0"])
    m["UA"] = mirror_frames(prep["UA"])
    m["UB"] = mirror_frames(prep["UB"])
    return m


# --------------------------------------------------------------------------- #
#  Наблюдаемые (батч)
# --------------------------------------------------------------------------- #

_tan_b = jax.vmap(B.node_tangents)
_etw_b = jax.vmap(lambda u, t: B.e_twist(u, t, 1.0))
_ta_b = jax.vmap(B.twist_angles)


def e_twist_mean(x, u):
    """Средняя по батчу твист-энергия (k_f=1). (B,N,4)×2 → скаляр float."""
    return float(_etw_b(u, _tan_b(x)).mean())


def twist_angle_samples(x, u):
    """Пул локальных твист-углов по батчу (для гистограмм/KS). → np.array (B·(N−1),)."""
    return np.asarray(_ta_b(u, _tan_b(x))).ravel()


def parity_all(x, u, UA, UB):
    """parity по батчу → np.array (B,) ±1."""
    return np.asarray(I.parity_batch(x, u, UA[0], UB[0]))


def twist_profile(x, u):
    """Профиль ⟨φ_i⟩ по позициям связи (усреднён по батчу) → np.array (N−1,).
    Показывает локализацию/делокализацию твист-солитона (V1)."""
    return np.asarray(_ta_b(u, _tan_b(x)).mean(0))


# --------------------------------------------------------------------------- #
#  Блочная релаксация (сходимость по наблюдаемой, §4.2 SPEC / CLAUDE.md)
# --------------------------------------------------------------------------- #

def relax_blocks_s3(stepper, prep, T, dt_steps, n_blocks, key, sample_last=5):
    """S³-ветка: n_blocks×dt_steps шагов при температуре T. Трек наблюдаемых по блокам.
    Пул локальных твист-углов по последним sample_last блокам (устойчивый KS в равновесии).
    Возврат dict: траектории e_twist, parity-доля(+1), rejected, финал x,u, ta_pool."""
    x, u = prep["x0"], prep["u0"]
    UA, UB = prep["UA"], prep["UB"]
    Tsched = jnp.full((dt_steps,), T)
    etw = [e_twist_mean(x, u)]
    par_frac = [float((parity_all(x, u, UA, UB) > 0).mean())]
    rej_total = 0.0
    ta_pool = []
    for blk in range(n_blocks):
        key, sk = jr.split(key)
        x, u, rej = stepper["run"](sk, x, u, prep["X0"], prep["XL"], UA, UB, Tsched)
        etw.append(e_twist_mean(x, u))
        par_frac.append(float((parity_all(x, u, UA, UB) > 0).mean()))
        rej_total += float(rej.sum())
        if blk >= n_blocks - sample_last:
            ta_pool.append(twist_angle_samples(x, u))
    return dict(x=x, u=u, e_twist=etw, parity_pos_frac=par_frac,
                rejected=rej_total, n_steps=n_blocks * dt_steps,
                ta_pool=np.concatenate(ta_pool) if ta_pool else twist_angle_samples(x, u))


def relax_blocks_so2(stepper, prep, T, dt_steps, n_blocks, key, W_target):
    """SO(2)-ветка: трек e_twist, winding W, rejected (ADD §8). Возврат dict."""
    x, phi = prep["x0"], prep["phi0"]
    UA, UB = prep["UA"], prep["UB"]
    Tsched = jnp.full((dt_steps,), T)
    etw = [e_twist_mean(x, B.u_of_phi(phi))]
    W = [float(B.winding(phi).mean())]
    rej_total = 0.0
    for _ in range(n_blocks):
        key, sk = jr.split(key)
        x, phi, rej = stepper["run_so2"](sk, x, phi, prep["X0"], prep["XL"], UA, UB,
                                         Tsched, W_target)
        etw.append(e_twist_mean(x, B.u_of_phi(phi)))
        W.append(float(B.winding(phi).mean()))
        rej_total += float(rej.sum())
    return dict(x=x, phi=phi, e_twist=etw, winding=W,
                rejected=rej_total, n_steps=n_blocks * dt_steps)


# --------------------------------------------------------------------------- #
#  Статистика
# --------------------------------------------------------------------------- #

def ks_two_sample(a, b):
    """Двухвыборочный KS (scipy). Возврат (statistic, pvalue)."""
    s = stats.ks_2samp(np.asarray(a), np.asarray(b))
    return float(s.statistic), float(s.pvalue)


def relaxation_halftime(etw, dt_steps, floor=None):
    """Грубое время релаксации: шаги до пересечения (E0+floor)/2 твист-энергией.
    floor — асимптота (по умолчанию последнее значение). Возврат шагов или None."""
    e = np.asarray(etw)
    fl = e[-1] if floor is None else floor
    target = 0.5 * (e[0] + fl)
    idx = np.where(e <= target)[0]
    return int(idx[0] * dt_steps) if len(idx) else None
