"""C2-F вторичная кампания T=0.05 (addendum п.4 «T=0.05 вторично»; новый prereg
не нужен). STAGED. Протокол S1: T_mid=0.05 + доохлаждение (как наука цикла 1).

Две правки readout против первичной T=0:
 (а) F_t — дальний конец читается ЕГО осью (b базового, b' повёрнутого) ⇒ нет
     degen-артефакта π/2;
 (б) диагностика A=|⟨s·t⟩| базового прогона (θ=0) — сверка с эталоном 0.418±0.026.
F_s (ближний, фиксированная ê) без изменений — фактор-наблюдаемая.

Сетка: k_f×1 Δγ×4 N∈{32,96}; k_f×4 Δγ∈{π/4,π/2} N=32; M=4800.
Kill-логика H-F1/H-F3 та же, вторичная. Сырьё→коммит→анализ.

Запуск: PYTHONPATH=src:phase_D:cycle2 python cycle2/run_c2f_campaign_t005.py --prereg-commit <hash>
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

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")

BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
LR = 5e-3
T_MID = 0.05
M_REPL = 8 if SMOKE else 4800
STEPS = 12 if SMOKE else 4000
REF = M.REF_AXIS
DGAMMA = {"pi/8": np.pi/8, "pi/4": np.pi/4, "3pi/8": 3*np.pi/8, "pi/2": np.pi/2}


def relax(mini, prep, key, a, b):
    """Протокол S1: T_mid=0.05 STEPS шагов + доохлаждение T=0 STEPS//2. Общий λ."""
    sched = jnp.concatenate([jnp.full((STEPS,), T_MID), jnp.zeros((STEPS // 2,))])
    _, u, _ = mini["run"](key, prep["x0"], prep["u0"], prep["X0"], prep["XL"], a, b, sched)
    return u


def read(u, a_read, b_read):
    """(s=sign(n_A·a_read), t=sign(n_B·b_read), degen)."""
    s, t, degen = M.classify_batch(u, a_read, b_read)
    return np.asarray(s), np.asarray(t), np.asarray(degen)


def frate(x, y):
    F = float(np.mean(x != y))
    return F, float(np.sqrt(max(F * (1 - F), 1e-9) / x.size))


def polar_prep(prep):
    nA = jax.vmap(lambda uu: M.end_axis(uu[0]))(prep["u0"])
    return np.asarray(jnp.arccos(jnp.clip(nA @ REF, -1.0, 1.0)))


def stratified_Fs(s_b, s_bp, theta):
    q = np.quantile(theta, [0.25, 0.5, 0.75]); bins = np.digitize(theta, q)
    out = []
    for k in range(4):
        m = bins == k; n = int(m.sum())
        out.append(dict(F=float(np.mean(s_b[m] != s_bp[m])) if n else 0.0, n=n))
    return out


def run_cell(kf, N, dgammas, key):
    params = dict(BASE, k_f=BASE["k_f"] * kf)
    mini = M.build_minimizer(params, lr=LR, freeze_w=False)
    prep = M.prep_dynamics(N, True, M_REPL, key)
    theta = polar_prep(prep)
    a, b0 = REF, REF
    am = -REF
    u_base = relax(mini, prep, key, a, b0)
    u_base_m = relax(mini, prep, key, am, -REF)
    # диагностика A=|⟨s·t⟩| базового (θ=0): читаем оба конца своей осью (ê,ê)
    sb0, tb0, db0 = read(u_base, a, b0)
    keep = ~db0
    st = (sb0 * tb0)[keep]
    A_stat = float(abs(np.mean(st))) if keep.sum() else 0.0
    A_sig = float(np.sqrt(max(1 - A_stat**2, 1e-9) / max(keep.sum(), 1)))
    res = dict(kf=kf, N=N, M=M_REPL,
               marginals=dict(P_s_plus=float(np.mean(sb0 > 0)), P_t_plus=float(np.mean(tb0 > 0)),
                              degen_base=float(np.mean(db0)), A_stat=A_stat, A_sigma_bin=A_sig),
               gamma={})
    for name, g in dgammas.items():
        _, bp = M.apparatus_axes_theta(g)
        u_p = relax(mini, prep, key, a, bp)
        # F_s: ближний, фиксированная ê оба
        sb, _, _ = read(u_base, a, REF); sp, _, _ = read(u_p, a, REF)
        Fs, sFs = frate(sb, sp)
        # F_t: дальний, ЕГО осью (base→b0=ê, prime→bp)  [правка (а)]
        _, tb, dtb = read(u_base, a, b0); _, tp, dtp = read(u_p, a, bp)
        Ft, sFt = frate(tb, tp)
        Ds = abs(float(np.mean(sb > 0)) - float(np.mean(sp > 0)))
        # зеркало ближнего
        u_pm = relax(mini, prep, key, am, -jnp.asarray(bp))
        sbm, _, _ = read(u_base_m, am, -REF); spm, _, _ = read(u_pm, am, -REF)
        Fs_m, _ = frate(sbm, spm)
        res["gamma"][name] = dict(
            Dgamma=float(g), F_s=Fs, sigma_Fs=sFs, F_t=Ft, sigma_Ft=sFt, Delta_s=Ds,
            degen_far_own=float(np.mean(dtp)), P_s_plus_bp=float(np.mean(sp > 0)),
            F_s_mirror=Fs_m, strat_Fs=stratified_Fs(sb, sp, theta))
        print(f"  k_f×{kf} N={N} Δγ={name:5s}: F_s={Fs:.4f}±{sFs:.4f} F_t(own)={Ft:.4f}±{sFt:.4f} "
              f"Δ_s={Ds:.4f} A={A_stat:.3f} degen_far={np.mean(dtp):.3f}")
    return res


def _require_approval():
    if "--prereg-commit" not in sys.argv:
        sys.exit("STAGED: запуск только с --prereg-commit <hash>.")
    return sys.argv[sys.argv.index("--prereg-commit") + 1]


def main():
    commit = _require_approval()
    dev = jax.devices()
    meta = dict(script="run_c2f_campaign_t005", prereg_commit=commit, addendum="eb4a7a8",
                backend=jax.default_backend(), jax_version=jax.__version__, device=str(dev[0]),
                x64=jax.config.jax_enable_x64, M=M_REPL, T_mid=T_MID, role="secondary T=0.05",
                readout="F_s: fixed ê; F_t: own axis (правка а)")
    print(f"[C2-F T=0.05] backend={meta['backend']} M={M_REPL} T_mid={T_MID}")
    out = {"meta": meta, "primary_kf1": {}, "confirming_kf4": {}}
    key = jr.PRNGKey(20260716 + 205)
    for N in (32, 96):
        key, sk = jr.split(key)
        out["primary_kf1"][f"N{N}"] = run_cell(1.0, N, DGAMMA, sk)
    key, sk = jr.split(key)
    conf = {k: DGAMMA[k] for k in ("pi/4", "pi/2")}
    out["confirming_kf4"]["N32"] = run_cell(4.0, 32, conf, sk)
    with open(os.path.join(RES, "C2F_campaign_T005_raw.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  СЫРЬЁ → {RES}/C2F_campaign_T005_raw.json (далее: КОММИТ, потом анализ)")


if __name__ == "__main__":
    main()
