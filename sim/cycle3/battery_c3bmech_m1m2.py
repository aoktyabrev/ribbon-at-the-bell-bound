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


def synth_ext(chi1, chi3, n=4_000_000):
    """Синтетич. ансамбль n~(1+χ₁n_z+χ₃n_z³) на S² (деформ. класс addendum3).
    Позитивность проверяется; при min<0 плотность клипуется (нота в отчёте)."""
    rng = np.random.default_rng(1)
    cc = np.linspace(-1, 1, 2001); dens = np.maximum(1 + chi1 * cc + chi3 * cc ** 3, 0.0)
    posmin = float((1 + chi1 * cc + chi3 * cc ** 3).min())
    mx = dens.max(); out = []; need = n
    while need > 0:
        v = rng.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        d = np.maximum(1 + chi1 * v[:, 2] + chi3 * v[:, 2] ** 3, 0.0)
        acc = rng.random(need * 2) < d / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n], posmin


def fit_ext(nm):
    """Фит расширенным классом (b): y=2p−1 = χ₁c + χ₃c³, взвеш. МНК на [c,c³]. χ²/dof."""
    c = nm[:, 2]
    hist, e = np.histogram(c, 20, (-1, 1), density=True); ctr = 0.5 * (e[:-1] + e[1:])
    y = 2 * hist - 1.0; W = 1.0 / (4 * np.maximum(hist, 1e-3) / (len(c) * 0.1))
    Xb = np.stack([ctr, ctr ** 3], 1)
    A = (Xb.T * W) @ Xb; bvec = (Xb.T * W) @ y
    coef = np.linalg.solve(A + 1e-9 * np.eye(2), bvec)
    resid = y - Xb @ coef
    chi2 = float(np.sum(resid ** 2 * W) / (20 - 2))
    return float(coef[0]), float(coef[1]), chi2


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
    print("\n M1-калибровка χ_frozen(rate) [квенч с полем]; фит РАСШИРЕННЫМ классом (b) 1+χ₁c+χ₃c³:")
    print(f"  {'rate':>6} {'χ_mom':>8} {'χ₁':>8} {'χ₃':>8} {'χ²/dof':>8} {'форма(b)':>9}")
    cal = {}; chosen = None
    for r in RATES:
        if r == 0:
            uf = up                                   # мгновенная заморозка: плато-конфиг as-is
        else:
            sf = jnp.linspace(T_PREP, 0.0, r)
            _, uf = relaxer(H)(jr.PRNGKey(9), xp, up, prep["X0"], prep["XL"], sf)
        nm = nmid_vec(uf); chi_mom = 3 * float(nm[:, 2].mean()); chi1, chi3, chi2 = fit_ext(nm)
        ok = chi2 < 2 and abs(chi_mom) > 3 * SIG
        cal[str(r)] = dict(chi_mom=chi_mom, chi1=chi1, chi3=chi3, chi2dof=chi2,
                           form_ok=bool(chi2 < 2), nm=nm)
        print(f"  {r:>6} {chi_mom:>8.4f} {chi1:>8.4f} {chi3:>8.4f} {chi2:>8.3f} {str(chi2<2):>9}")
        if chosen is None and ok:
            chosen = r
    # выбор ансамбля: (b) самая быстрая с формой; иначе (b′) form-free = самая быстрая с D>0
    form_free = False
    if chosen is None:
        for r in RATES:
            if abs(cal[str(r)]["chi_mom"]) > 3 * SIG:
                chosen = r; form_free = True; break
    if chosen is None:
        print("  СТЕНА (c): D_mech<3σ на всех ступенях ⇒ источник не поляризуется. Фронт A стоп.")
        json.dump(dict(calibration={k: {kk: vv for kk, vv in v.items() if kk != 'nm'}
                                     for k, v in cal.items()}, wall_c=True),
                  open(os.path.join(RES, f"C3Bmech_M1M2{'_smoke' if SMOKE else ''}.json"), "w"), indent=2)
        return
    mode = "(b′) form-free" if form_free else "(b) расширенный класс"
    print(f"  ВЫБРАНА ступень rate={chosen}, режим клейма: {mode}")
    nm = cal[str(chosen)]["nm"]; chi1, chi3 = cal[str(chosen)]["chi1"], cal[str(chosen)]["chi3"]
    chi_f = cal[str(chosen)]["chi_mom"]

    # --- контроль чистоты h=0 через тот же fast-freeze ---
    _, xp0, up0 = get_source(0.0)
    if chosen == 0:
        uf0 = up0
    else:
        _, uf0 = relaxer(0.0)(jr.PRNGKey(9), xp0, up0, prep["X0"], prep["XL"], jnp.linspace(T_PREP, 0.0, chosen))
    nm0 = nmid_vec(uf0)

    # --- M1: D_mech (3-й И 5-й, стандарт A3) vs аналитика на ФИТ. мере (b) ---
    d3, d5, dmean = D_mech(nm); d3_0, d5_0, dmean0 = D_mech(nm0)
    syn, posmin = synth_ext(chi1, chi3); d3_an, d5_an, _ = D_mech(syn)
    print(f"\n M1 — стиринг D_mech vs аналитика (синт. 1+{chi1:.3f}c+{chi3:.3f}c³; min-плотн={posmin:.3f}):")
    print(f"  D3_мех={d3:.5f} (аналит {d3_an:.5f})  D5_мех={d5:.5f} (аналит {d5_an:.5f})  |Δmean|={dmean:.5f}")
    print(f"  КОНТРОЛЬ h=0: D3={d3_0:.5f} D5={d5_0:.5f} (<2σ={2*SIG:.4f}: {d3_0<2*SIG})  |Δmean|={dmean0:.5f}")
    match3 = abs(d3 - d3_an) < max(3 * SIG, 0.15 * abs(d3_an))
    match5 = abs(d5 - d5_an) < max(3 * SIG, 0.20 * abs(d5_an) + 1e-9)
    m1_pass = (d3 > 5 * SIG) and match3 and match5 and (d3_0 < 2 * SIG)
    print(f"  матч 3-й={match3} 5-й={match5}; M1 ВЕРДИКТ: {'ПРОХОД (D>0, матч 3+5, h=0 чист)' if m1_pass else 'разбор'}")

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

    full_pass = m1_pass and m2_pass
    # вердикт клейм-строки (addendum3): вставлена при проходе full; иначе (b′)/стена
    if full_pass and not form_free:
        claim = "ВСТАВЛЕНА клейм-строка (b) — премиса отбора на упругой механике ЦЕЛИКОМ"
    elif full_pass and form_free:
        claim = "(b′) form-free: премиса = полярная мера с D>0 (форма не фитится, Йенсен-на-эмпирике)"
    else:
        claim = "НЕ вставлена: разбор (M1/M2 не прошли на full)"
    tag = "_smoke" if SMOKE else ""
    json.dump(dict(calibration={k: {kk: vv for kk, vv in v.items() if kk != 'nm'} for k, v in cal.items()},
                   chosen_rate=chosen, mode=mode, chi_frozen=chi_f, chi1=chi1, chi3=chi3, posmin=posmin,
                   M1=dict(d3=d3, d5=d5, dmean=dmean, d3_analytic=d3_an, d5_analytic=d5_an,
                           d3_h0=d3_0, d5_h0=d5_0, dmean_h0=dmean0, pass_=bool(m1_pass)),
                   M2=dict(scan=scan, zero_num=zero_num, zero_analytic=zero_an, max_mismatch=max_mism, pass_=bool(m2_pass)),
                   full_pass=bool(full_pass), form_free=bool(form_free), claim_verdict=claim,
                   sigma=SIG, N=N, batch=BATCH), open(os.path.join(RES, f"C3Bmech_M1M2{tag}.json"), "w"), indent=2)
    print(f"\n === ФРОНТ A: M1 {'✓' if m1_pass else '✗'} M2 {'✓' if m2_pass else '✗'} ({mode}) ⇒ {claim} ===")
    print(f"  → {RES}/C3Bmech_M1M2{tag}.json")


if __name__ == "__main__":
    main()
