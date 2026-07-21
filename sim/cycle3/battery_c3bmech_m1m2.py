"""C3-B-mech / M1+M2 (prereg 63456cd). Fast-freeze источник (плато T_prep, поле
h=0.05 → БЫСТРЫЙ квенч T→0 С ПОЛЕМ → поле off). Калибровка χ_frozen(rate);
M1 стиринг D_mech vs аналитика (матч. форма 1+χc); M2 телеграф Δ(p) нуль-множ.;
контроль чистоты h=0. GPU freeze + numpy анализ. --smoke.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import json
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

_SIM = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_SIM, "phase_D")); sys.path.insert(0, os.path.join(_SIM, "src"))
import band4d as B
import measurement as M
from ribbon_sim.frames import normalize

SMOKE = "--smoke" in sys.argv
RES = os.path.join(_SIM, "phase_D", "results")
PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, ell=1.0)
N = 24 if SMOKE else 32
BATCH = 2000 if SMOKE else 20000
LR, T_HOT, T_PREP, H = 5e-3, 2.0, 0.5, 0.05
RATES = [0, 100, 300, 600, 1200]   # rate=0 = мгновенная заморозка (плато-конфиг as-is)
P_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, np.inf]
Z = np.array([0.0, 0.0, 1.0]); X = np.array([1.0, 0.0, 0.0])
SIG = 1.0 / np.sqrt(BATCH)
axis_all = jax.vmap(M.end_axis)


def e_prep(x, u, h):
    t = B.node_tangents(x)
    nz = axis_all(u)[:, 2]
    return (B.e_stretch(x, PARAMS["k_s"], PARAMS["ell"]) + B.e_bend(t, PARAMS["k_b"])
            + B.e_twist(u, t, PARAMS["k_f"]) - h * jnp.sum(nz))


def relaxer(h):
    gx = jax.vmap(jax.grad(lambda x, u: e_prep(x, u, h), 0))
    gu = jax.vmap(jax.grad(lambda x, u: e_prep(x, u, h), 1))

    @jax.jit
    def run(key, x0, u0, X0, XL, T_sched):
        def step(carry, inp):
            x, u = carry; T, k = inp
            kx, ku = jr.split(k)
            x = (x - LR * gx(x, u) + jr.normal(kx, x.shape) * jnp.sqrt(2 * LR * T)).at[:, 0].set(X0).at[:, -1].set(XL)
            du = -LR * gu(x, u) + jr.normal(ku, u.shape) * jnp.sqrt(2 * LR * T)
            du = du - jnp.sum(du * u, -1, keepdims=True) * u
            u = normalize(u + du)
            u = u * jnp.sign(jnp.sum(u * u, -1, keepdims=True) + 1e-30)
            return (x, u), None
        keys = jr.split(key, T_sched.shape[0])
        (x, u), _ = jax.lax.scan(step, (x0, u0), (T_sched, keys))
        return x, u
    return run


def nmid_vec(u):
    lo, hi = N // 3, 2 * N // 3
    ax = jax.vmap(axis_all)(u)[:, lo:hi, :].mean(1)
    return np.asarray(ax / (jnp.linalg.norm(ax, axis=-1, keepdims=True) + 1e-12))


def f_p(c, p):
    if np.isinf(p):
        return np.where(c > 0, 1.0, np.where(c < 0, 0.0, 0.5))
    A = np.abs(1 + c) ** (p / 2); Bb = np.abs(1 - c) ** (p / 2)
    return A / (A + Bb)


def mom(nm, a, k):
    proj = nm @ a; m = 0.0
    for mask in (proj >= 0, proj < 0):
        w = mask.mean(); nb = nm[mask].mean(0); m += w * (nb @ Z) ** k
    return m


def D_mech(nm):
    dmean = abs(mom(nm, Z, 1) - mom(nm, X, 1))
    d3 = abs(mom(nm, Z, 3) - mom(nm, X, 3)); d5 = abs(mom(nm, Z, 5) - mom(nm, X, 5))
    return d3, d5, dmean


def telegraph(nm, p):
    def bobP(a):
        proj = nm @ a; P = 0.0
        for mask in (proj >= 0, proj < 0):
            w = mask.mean(); nb = nm[mask].mean(0); P += w * f_p(nb @ Z, p)
        return P
    return abs(bobP(Z) - bobP(X))


def synth_mu(chi, n=4_000_000):
    """Синтетический ансамбль n~(1+χ n_z) на S² — аналитика матч. формы."""
    rng = np.random.default_rng(1); mx = 1 + abs(chi); out = []; need = n
    while need > 0:
        v = rng.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = rng.random(need * 2) < (1 + chi * v[:, 2]) / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def fit_chi(nm):
    c = nm[:, 2]; chi = 3 * float(c.mean())
    hist, e = np.histogram(c, 20, (-1, 1), density=True); ctr = 0.5 * (e[:-1] + e[1:])
    y = 2 * hist - 1; var = 4 * np.maximum(hist, 1e-3) / (len(c) * 0.1)
    cf = float(np.sum(ctr * y / var) / (np.sum(ctr * ctr / var) + 1e-12))
    chi2 = float(np.sum((y - cf * ctr) ** 2 / var) / 19)
    return chi, cf, chi2


def get_source(h):
    prep = M.prep_dynamics(N, False, BATCH, jr.PRNGKey(20260721))
    ramp = 400 if SMOKE else 3000; plat = 300 if SMOKE else 2000
    sp = jnp.concatenate([jnp.linspace(T_HOT, T_PREP, ramp), jnp.full(plat, T_PREP)])
    x, u = relaxer(h)(jr.PRNGKey(7), prep["x0"], prep["u0"], prep["X0"], prep["XL"], sp)
    return prep, x, u


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N} BATCH={BATCH} T_prep={T_PREP} h={H} GPU={jax.devices()}")
    prep, xp, up = get_source(H)
    # --- калибровка χ_frozen(rate): БЫСТРЫЙ квенч С ПОЛЕМ ---
    print("\n M1-калибровка χ_frozen(rate) [квенч T_prep→0 с полем h]:")
    print(f"  {'rate':>6} {'χ_frozen':>10} {'фит χ':>8} {'χ²/dof':>8} {'форма':>6}")
    cal = {}; chosen = None
    for r in RATES:
        if r == 0:
            uf = up                                   # мгновенная заморозка: плато-конфиг as-is
        else:
            sf = jnp.linspace(T_PREP, 0.0, r)
            _, uf = relaxer(H)(jr.PRNGKey(9), xp, up, prep["X0"], prep["XL"], sf)
        nm = nmid_vec(uf); chi, cf, chi2 = fit_chi(nm)
        ok = chi2 < 2 and abs(chi) > 3 * SIG
        cal[str(r)] = dict(chi_frozen=chi, chi_fit=cf, chi2dof=chi2, form_ok=bool(chi2 < 2), nm=nm)
        print(f"  {r:>6} {chi:>10.4f} {cf:>8.4f} {chi2:>8.3f} {str(chi2<2):>6}")
        if chosen is None and ok:
            chosen = r
    if chosen is None:
        print("  KILL протокола: ни одна ступень не держит χ_frozen>3σ с валидной формой ⇒")
        print("    СТЕНА «биас источника не переживает заморозку». Фронт A стопорится.")
        json.dump(dict(calibration={k: {kk: vv for kk, vv in v.items() if kk != 'nm'}
                                     for k, v in cal.items()}, wall=True),
                  open(os.path.join(RES, f"C3Bmech_M1M2{'_smoke' if SMOKE else ''}.json"), "w"), indent=2)
        return
    print(f"  ВЫБРАНА ступень (самая быстрая, форма ok): rate={chosen}")
    nm = cal[str(chosen)]["nm"]; chi_f = cal[str(chosen)]["chi_frozen"]

    # --- контроль чистоты h=0 через тот же fast-freeze ---
    _, xp0, up0 = get_source(0.0)
    if chosen == 0:
        uf0 = up0
    else:
        _, uf0 = relaxer(0.0)(jr.PRNGKey(9), xp0, up0, prep["X0"], prep["XL"], jnp.linspace(T_PREP, 0.0, chosen))
    nm0 = nmid_vec(uf0)

    # --- M1: D_mech vs аналитика (синтетич. μ_χ при χ_f) ---
    d3, d5, dmean = D_mech(nm); d3_0, d5_0, dmean0 = D_mech(nm0)
    syn = synth_mu(chi_f); d3_an, d5_an, _ = D_mech(syn)
    print("\n M1 — стиринг D_mech (механич.) vs аналитика (синтетич. 1+χ_f·c):")
    print(f"  χ_frozen={chi_f:.4f}  D_mech(3)={d3:.5f} (аналит {d3_an:.5f})  D5={d5:.5f}  |Δmean|={dmean:.5f}")
    print(f"  КОНТРОЛЬ h=0: D_mech(3)={d3_0:.5f} (<2σ={2*SIG:.4f}: {d3_0<2*SIG})  |Δmean|={dmean0:.5f}")
    m1_pass = (d3 > 5 * SIG) and (abs(d3 - d3_an) < max(3 * SIG, 0.15 * d3_an)) and (d3_0 < 2 * SIG)
    print(f"  M1 ВЕРДИКТ: {'ПРОХОД (D_mech>0, матч аналитике, h=0 чист)' if m1_pass else 'разбор'}")

    # --- M2: телеграф-скан Δ(p) на механич. ансамбле ---
    print("\n M2 — телеграф Δ(p) на механич. условных ансамблях (арбитр = аналитика):")
    syn_sig = 1.0 / np.sqrt(len(syn)); scan = {}; zero_num = []; zero_an = []; max_mism = 0.0
    for p in P_GRID:
        dnum = telegraph(nm, p); dan = telegraph(syn, p); key = "inf" if np.isinf(p) else f"{p}"
        scan[key] = dict(num=float(dnum), analytic=float(dan), mism=float(abs(dnum - dan)))
        max_mism = max(max_mism, abs(dnum - dan))
        if dnum < 2 * SIG:
            zero_num.append(key)
        if dan < 2 * syn_sig:
            zero_an.append(key)
        print(f"  p={key:>4}: Δ_мех={dnum:.6f} аналит={dan:.6f} |ан−чис|={abs(dnum-dan):.6f} "
              f"{'(числ=0)' if dnum<2*SIG else ''}")
    # арбитр: аналит. нуль-множ.={2} ∧ числ↔аналит согласие <2σ (стандарт L3/hardening)
    m2_pass = (zero_an == ["2.0"]) and (max_mism < 2 * SIG)
    print(f"  нуль-множ. числ={zero_num} аналит={zero_an}; max|ан−чис|={max_mism:.5f} (2σ={2*SIG:.4f})")
    print(f"  M2 ВЕРДИКТ: {'ПРОХОД (аналит={2}, согласие <2σ)' if m2_pass else 'разбор'}")

    tag = "_smoke" if SMOKE else ""
    json.dump(dict(calibration={k: {kk: vv for kk, vv in v.items() if kk != 'nm'} for k, v in cal.items()},
                   chosen_rate=chosen, chi_frozen=chi_f,
                   M1=dict(d3=d3, d5=d5, dmean=dmean, d3_analytic=d3_an, d3_h0=d3_0, dmean_h0=dmean0, pass_=bool(m1_pass)),
                   M2=dict(scan=scan, zero_set=zero, pass_=bool(m2_pass)),
                   sigma=SIG, N=N, batch=BATCH), open(os.path.join(RES, f"C3Bmech_M1M2{tag}.json"), "w"), indent=2)
    print(f"\n === ФРОНТ A: M1 {'✓' if m1_pass else '✗'} M2 {'✓' if m2_pass else '✗'} "
          f"⇒ {'премиса отбора на упругой механике ЦЕЛИКОМ' if (m1_pass and m2_pass) else 'не замкнуто'} ===")
    print(f"  → {RES}/C3Bmech_M1M2{tag}.json")


if __name__ == "__main__":
    main()
