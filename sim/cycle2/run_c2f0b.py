"""C2-F0b — позитив-контроль харнеса (см. C2F_prereg_addendum1.md п.2).
STAGED: запуск только с --prereg-commit после утверждения (addendum eb4a7a8).

Гейт инструмента: поворот СВОЕГО зажима конца на π/2 обязан флипать знак
этого конца (читаем ФИКСИРОВАННОЙ осью ê обоих прогонов, общий λ).
  F_s^self: a→a' (π/2), b=ê фикс.; читаем ближний знак sign(n_A·ê).
  F_t^self: b→b' (π/2), a=ê фикс.; читаем дальний знак sign(n_B·ê).
Оба ≫0 ⇒ харнес видит флипы, b' доходит до вычисления ⇒ F_s≡0 в F0 —
свойство динамики. KILL: любой из двух = 0 ⇒ стоп-аудит; F0 неинформативен.

Замечание readout: читаем ОБА прогона одной осью ê (не «своей» осью каждого),
иначе каждый конец тривиально выровнен к своему зажиму и флип не виден.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2f0b.py --prereg-commit <hash>
Бэкенд GPU/fp64.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR, N, M_REPL, KF_MULT = 5e-3, 32, 1200, 4.0
STEPS = 4000
REF = M.REF_AXIS                       # ê=(0,0,1), фиксированная ось считывания


def relax(mini, prep, key, a, b):
    """Общий λ: тот же prep/key, зажимы (a,b), квенч T=0. Возврат u (M,N,4)."""
    sched = jnp.concatenate([jnp.zeros((STEPS,)), jnp.zeros((STEPS // 2,))])
    _, u, _ = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u


def near_sign(u):
    """sign(n_A·ê) — ближний конец, читаем фиксированной ê."""
    s, _, _ = M.classify_batch(u, REF, REF)
    return np.asarray(s)


def far_sign(u):
    """sign(n_B·ê) — дальний конец, читаем фиксированной ê."""
    _, t, _ = M.classify_batch(u, REF, REF)
    return np.asarray(t)


def frate(x, y):
    F = float(np.mean(x != y))
    return F, float(np.sqrt(max(F * (1 - F), 1e-9) / x.size))


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash> после утверждения Артемом.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2f0b", prereg_commit=commit, backend=jax.default_backend(),
                jax_version=jax.__version__, device=str(dev[0]), x64=jax.config.jax_enable_x64,
                N=N, M=M_REPL, kf_mult=KF_MULT, T=0.0)
    print(f"[C2-F0b] backend={meta['backend']} device={meta['device']} k_f×{KF_MULT}")
    params = dict(BASE, k_f=BASE["k_f"] * KF_MULT)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    key = jr.PRNGKey(20260716 + 41)
    prep = M.prep_dynamics(N, True, M_REPL, key)

    a = REF                                     # ê
    a_prime, b_prime = np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])   # π/2 от ê в (x,z)
    a_prime = jnp.asarray(a_prime); b_prime = jnp.asarray(b_prime)

    # базовый прогон (a=ê, b=ê) — общий для обоих контролей
    u0 = relax(mini, prep, key, a, a)
    # контроль ближнего: поворот a→a', b=ê фикс.
    u_near = relax(mini, prep, key, a_prime, a)
    # контроль дальнего: поворот b→b', a=ê фикс.
    u_far = relax(mini, prep, key, a, b_prime)

    Fs, sFs = frate(near_sign(u0), near_sign(u_near))
    Ft, sFt = frate(far_sign(u0), far_sign(u_far))
    Ps0 = float(np.mean(near_sign(u0) > 0)); Pt0 = float(np.mean(far_sign(u0) > 0))

    kill = (Fs == 0.0) or (Ft == 0.0)
    out = dict(meta=meta,
               F_s_self=dict(F=Fs, sigma=sFs, rot="a→a' π/2, b=ê"),
               F_t_self=dict(F=Ft, sigma=sFt, rot="b→b' π/2, a=ê"),
               marginals=dict(P_s_plus=Ps0, P_t_plus=Pt0),
               kill=bool(kill))
    print(f"  F_s^self (a→a' π/2) = {Fs:.4f}±{sFs:.4f}   [{'KILL' if Fs==0 else 'жив'}]")
    print(f"  F_t^self (b→b' π/2) = {Ft:.4f}±{sFt:.4f}   [{'KILL' if Ft==0 else 'жив'}]")
    print(f"  маргиналы: P(s=+)={Ps0:.3f}  P(t=+)={Pt0:.3f}")
    with open(os.path.join(RES, "C2F0b_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2F0b_raw.json  ГЕЙТ F0b: {'KILL — стоп-аудит харнеса' if kill else 'ПРОХОД'}")
    if kill:
        sys.exit(1)


if __name__ == "__main__":
    main()
