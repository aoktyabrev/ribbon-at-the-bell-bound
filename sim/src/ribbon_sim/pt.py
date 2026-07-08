"""Параллельное темперирование (replica exchange) для глассовых ландшафтов (R4b-PT).

Ядро (обмен реплик + лестница + диагностика) параметризовано energy_fn и step_fn,
чтобы логику обмена тестировать на игрушечной двухъямке независимо от S³-динамики.

Состояние: (B, R, *state_dims) — B лент, R реплик на ленту (по лестнице температур).
Обмен соседних реплик по Метрополису: accept с вероятностью
min(1, exp((β_i − β_j)(E_i − E_j))), детальный баланс (Katzgraber et al.).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def geometric_ladder(T_hot, T_cold, R):
    """Геометрическая лестница R температур от T_hot до T_cold (убывающая β)."""
    return jnp.geomspace(T_hot, T_cold, R)


def _swap_sweep(state, E, betas, key, parity, perm, tag, rt):
    """Одна развёртка обменов чётности parity: пары (r,r+1), r ≡ parity (mod 2).

    Обменивает конфигурации соседних реплик для принятых лент; тянет за собой E,
    perm (позиция→уокер), обновляет tag/rt (round-trips). Возврат обновлённых.
    """
    R = betas.shape[0]
    B = state.shape[0]
    acc_counts = jnp.zeros(R - 1)
    keys = jax.random.split(key, R)  # по ключу на пару (с запасом)
    for r in range(parity, R - 1, 2):
        dE = E[:, r] - E[:, r + 1]                       # (B,)
        delta = (betas[r] - betas[r + 1]) * dE           # (B,)
        u = jax.random.uniform(keys[r], (B,))
        accept = u < jnp.exp(jnp.minimum(delta, 0.0))    # min(1,exp) Метрополис
        a3 = accept[:, None, None]
        while a3.ndim < state.ndim:
            a3 = a3[..., None]
        sr, sr1 = state[:, r], state[:, r + 1]
        state = state.at[:, r].set(jnp.where(a3[:, 0], sr1, sr))
        state = state.at[:, r + 1].set(jnp.where(a3[:, 0], sr, sr1))
        Er, Er1 = E[:, r], E[:, r + 1]
        E = E.at[:, r].set(jnp.where(accept, Er1, Er))
        E = E.at[:, r + 1].set(jnp.where(accept, Er, Er1))
        pr, pr1 = perm[:, r], perm[:, r + 1]
        perm = perm.at[:, r].set(jnp.where(accept, pr1, pr))
        perm = perm.at[:, r + 1].set(jnp.where(accept, pr, pr1))
        acc_counts = acc_counts.at[r].add(jnp.sum(accept))
    return state, E, perm, tag, rt, acc_counts


def _update_roundtrips(perm, tag, rt):
    """Round-trips: уокер на холодном конце (поз.0) с tag=hot ⇒ +1 round-trip, tag←cold;
    на горячем (поз.R-1) ⇒ tag←hot. tag: 0 none,1 cold,2 hot. rt индексирован по уокеру."""
    B, R = perm.shape
    bidx = jnp.arange(B)
    w_cold = perm[:, 0]           # уокер на холодном конце
    w_hot = perm[:, R - 1]        # уокер на горячем конце
    tag_cold = tag[bidx, w_cold]
    completed = tag_cold == 2     # был на горячем ⇒ пришёл на холодный = round-trip
    rt = rt.at[bidx, w_cold].add(completed.astype(rt.dtype))
    tag = tag.at[bidx, w_cold].set(1)
    tag = tag.at[bidx, w_hot].set(2)
    return tag, rt


def make_pt(energy_fn, step_fn, betas, swap_every):
    """Собирает jit-прогон PT.

    energy_fn(state) → (B,R); step_fn(state, T_vec, key) — один ланжевен-шаг всех реплик
    (T_vec形 (R,) — температура реплики). Возврат run(key, state0, n_blocks) →
    (state_final, diag) где diag: acceptance по звеньям (R-1,), min/mean round-trips.
    """
    R = betas.shape[0]
    Ts = 1.0 / betas

    @partial(jax.jit, static_argnums=(2,))
    def run(key, state0, n_blocks):
        B = state0.shape[0]
        perm0 = jnp.broadcast_to(jnp.arange(R), (B, R))
        tag0 = jnp.zeros((B, R), dtype=jnp.int32).at[:, 0].set(1)  # холодный уокер начат
        rt0 = jnp.zeros((B, R), dtype=jnp.int32)
        acc0 = jnp.zeros(R - 1)

        def block(carry, blk):
            state, key, perm, tag, rt, acc = carry
            # swap_every ланжевен-шагов
            def lstep(s, k):
                return step_fn(s, Ts, k), None
            key, sub = jax.random.split(key)
            steps_keys = jax.random.split(sub, swap_every)
            state, _ = lax.scan(lstep, state, steps_keys)
            # обмен (чётность чередуется по блокам)
            key, sk = jax.random.split(key)
            E = energy_fn(state)
            parity = blk % 2
            # обе чётности гарантируют покрытие всех звеньев за 2 блока;
            # берём parity текущего блока
            state, E, perm, tag, rt, acc_c = lax.switch(
                parity,
                [lambda: _swap_sweep(state, E, betas, sk, 0, perm, tag, rt),
                 lambda: _swap_sweep(state, E, betas, sk, 1, perm, tag, rt)],
            )
            tag, rt = _update_roundtrips(perm, tag, rt)
            acc = acc + acc_c
            return (state, key, perm, tag, rt, acc), None

        (state, key, perm, tag, rt, acc), _ = lax.scan(
            block, (state0, key, perm0, tag0, rt0, acc0), jnp.arange(n_blocks))
        # acceptance-доля по звеньям: acc / (число попыток на звено)
        # звено r обменивается когда parity==r%2, т.е. в половине блоков
        attempts = jnp.where(jnp.arange(R - 1) % 2 == 0,
                             (n_blocks + 1) // 2, n_blocks // 2) * B
        accept_frac = acc / jnp.maximum(attempts, 1)
        rt_per_walker = jnp.min(rt, axis=1)  # мин round-trip по уокерам ленты
        return state, {"accept_frac": accept_frac,
                       "min_roundtrips": jnp.min(rt),
                       "mean_roundtrips": jnp.mean(rt),
                       "rt_per_ribbon_min": rt_per_walker}

    return run
