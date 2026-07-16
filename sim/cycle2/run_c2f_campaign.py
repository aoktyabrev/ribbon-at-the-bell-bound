"""C2-F основная кампания (см. C2F_prereg.md + addendum1 п.4). STAGED.
Запуск только с --prereg-commit после прохода F0b (e70e658).

Первичная ячейка k_f×1 (Δγ∈{π/8,π/4,3π/8,π/2}, N∈{32,96}, M=4800, T=0);
подтверждающая k_f×4 (Δγ∈{π/4,π/2}, N=32). Общий λ (prep+key), варьируем
удалённый зажим b→b'(Δγ). a=ê фикс. Считывание — фиксированной ê (оба конца).

Наблюдаемые (addendum): F_s (ближний, фактор-тест), F_t (дальний, симметричный
партнёр + liveness), Δ_s маргинал; маргиналы P(s=+),P(t=+); стратификация F_s
по квартилям начального полярного угла arccos(n_A·ê) на prep. Зеркала (a,b)→(−a,−b).
DEGENERATE диагностикой (sign(0)→+1, не отбрасывается). Сырьё→коммит→анализ.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2f_campaign.py --prereg-commit <hash>
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

SMOKE = "--smoke" in sys.argv       # только исполняемость (M≤8), статистики НЕ снимать
BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR = 5e-3
M_REPL = 8 if SMOKE else 4800
STEPS = 12 if SMOKE else 4000
REF = M.REF_AXIS
DGAMMA = {"pi/8": np.pi/8, "pi/4": np.pi/4, "3pi/8": 3*np.pi/8, "pi/2": np.pi/2}


def relax(mini, prep, key, a, b):
    """Общий λ, зажимы (a,b), квенч T=0. Возврат u (M,N,4)."""
    sched = jnp.zeros((STEPS + STEPS // 2,))
    _, u, _ = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u


def signs(u):
    """(s=sign(n_A·ê), t=sign(n_B·ê), degen) — оба конца фиксированной ê."""
    s, t, degen = M.classify_batch(u, REF, REF)
    return np.asarray(s), np.asarray(t), np.asarray(degen)


def frate(x, y):
    F = float(np.mean(x != y))
    return F, float(np.sqrt(max(F * (1 - F), 1e-9) / x.size))


def polar_prep(prep):
    """Начальный полярный угол ближнего конца arccos(n_A·ê) на prep (M,)."""
    nA = jax.vmap(lambda uu: M.end_axis(uu[0]))(prep["u0"])
    return np.asarray(jnp.arccos(jnp.clip(nA @ REF, -1.0, 1.0)))


def stratified_Fs(s_b, s_bp, theta):
    """F_s по квартилям θ_prep. Возврат list из 4 (F, n)."""
    q = np.quantile(theta, [0.25, 0.5, 0.75])
    bins = np.digitize(theta, q)
    out = []
    for k in range(4):
        m = bins == k
        n = int(m.sum())
        F = float(np.mean(s_b[m] != s_bp[m])) if n else 0.0
        out.append(dict(F=F, n=n))
    return out


def run_cell(kf, N, dgammas, key):
    """Одна ячейка (kf,N): база + лестница Δγ + зеркало. Возврат dict."""
    params = dict(BASE, k_f=BASE["k_f"] * kf)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    prep = M.prep_dynamics(N, True, M_REPL, key)
    theta = polar_prep(prep)
    a, b0 = REF, REF
    am, b0m = -REF, -REF
    u_base = relax(mini, prep, key, a, b0)
    u_base_m = relax(mini, prep, key, am, b0m)
    s0, t0, d0 = signs(u_base)
    res = dict(kf=kf, N=N, M=M_REPL,
               marginals=dict(P_s_plus=float(np.mean(s0 > 0)),
                              P_t_plus=float(np.mean(t0 > 0)),
                              degen_base=float(np.mean(d0))),
               gamma={})
    for name, g in dgammas.items():
        _, bp = M.apparatus_axes_theta(g)
        u_p = relax(mini, prep, key, a, bp)
        sb, tb, db = signs(u_base); sp, tp, dp = signs(u_p)
        Fs, sFs = frate(sb, sp)
        Ft, sFt = frate(tb, tp)
        Ds = abs(float(np.mean(sb > 0)) - float(np.mean(sp > 0)))
        # зеркало
        _, bpm = M.apparatus_axes_theta(g)
        u_pm = relax(mini, prep, key, am, -jnp.asarray(bpm))
        sbm, _, _ = signs(u_base_m); spm, _, _ = signs(u_pm)
        Fs_m, _ = frate(sbm, spm)
        res["gamma"][name] = dict(
            Dgamma=float(g), F_s=Fs, sigma_Fs=sFs, F_t=Ft, sigma_Ft=sFt,
            Delta_s=Ds, degen_diag=float(np.mean(dp)),
            P_s_plus_bp=float(np.mean(sp > 0)), P_t_plus_bp=float(np.mean(tp > 0)),
            F_s_mirror=Fs_m, strat_Fs=stratified_Fs(sb, sp, theta))
        print(f"  k_f×{kf} N={N} Δγ={name:5s}: F_s={Fs:.4f}±{sFs:.4f} F_t={Ft:.4f}±{sFt:.4f} "
              f"Δ_s={Ds:.4f} F_s^mir={Fs_m:.4f} degen={np.mean(dp):.3f}")
    return res


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash> после прохода F0b.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2f_campaign", prereg_commit=commit, addendum="eb4a7a8",
                backend=jax.default_backend(), jax_version=jax.__version__,
                device=str(dev[0]), x64=jax.config.jax_enable_x64, M=M_REPL, T=0.0,
                readout="fixed ê both ends")
    print(f"[C2-F campaign] backend={meta['backend']} device={meta['device']} M={M_REPL} T=0")
    out = {"meta": meta, "primary_kf1": {}, "confirming_kf4": {}}
    key = jr.PRNGKey(20260716 + 100)

    # первичная k_f×1: полная сетка, N∈{32,96}
    for N in (32, 96):
        key, sk = jr.split(key)
        out["primary_kf1"][f"N{N}"] = run_cell(1.0, N, DGAMMA, sk)
    # подтверждающая k_f×4: Δγ∈{π/4,π/2}, N=32
    key, sk = jr.split(key)
    conf = {k: DGAMMA[k] for k in ("pi/4", "pi/2")}
    out["confirming_kf4"]["N32"] = run_cell(4.0, 32, conf, sk)

    with open(os.path.join(RES, "C2F_campaign_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2F_campaign_raw.json (далее: КОММИТ, потом анализ H-F1/2/3)")


if __name__ == "__main__":
    main()
