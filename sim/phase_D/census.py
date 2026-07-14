"""D1-A. Топологическая перепись (СТАТИКА) — см. D1_prereg.md §D-H3a, аддендум §4.

Для каждой ячейки {сектор ∈ чёт/нечёт}×{ветвь (s,t)∈{±}²}×{θ∈{0,π}}: существует ли
НЕсингулярная конфигурация, совместимая с зажатыми осевыми ветвями обеих сторон?
Вердикты: EXISTS / NOT-FOUND / DEGENERATE. NOT-FOUND ≠ доказательство запрета.

Метод: constrained-минимизация E (осевые зажимы) из ≥20 разнообразных стартов +
адиабатическое протаскивание по θ из соседней ячейки. Реплика «в ячейке» если ось
резко даёт (s,t), сектор сохранён (ноль пересечений стены), минимум несингулярен.

Запуск:  JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/census.py [--smoke]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import itertools
import json
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import band4d as B
import invariant as I
import measurement as M

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
NSTARTS = 12 if SMOKE else 24
STEPS = 3000 if SMOKE else 5000
N = 16 if SMOKE else 48
LR = 5e-3


def run_cell(sector_odd, s, t, theta_pi, key, drag_from=None):
    """Минимизация ячейки из NSTARTS разнообразных стартов (+опц. адиаб. протаскивание).
    Возврат dict со статистикой и вердиктом."""
    a, b = M.apparatus_axes(theta_pi)
    mini = M.build_minimizer(PARAMS, lr=LR, freeze_w=False)

    # разнообразные старты: варьируем σ_frame по репликам
    keys = jr.split(key, NSTARTS)
    sigmas = np.linspace(0.0, 0.12, NSTARTS)
    xs, us, X0s, XLs = [], [], [], []
    for i in range(NSTARTS):
        p = M.prep_sector(N, sector_odd, s, t, theta_pi, 1, keys[i],
                          sigma_pos=0.05, sigma_frame=float(sigmas[i]))
        xs.append(p["x0"]); us.append(p["u0"]); X0s.append(p["X0"]); XLs.append(p["XL"])
    x0 = jnp.concatenate(xs); u0 = jnp.concatenate(us)
    X0 = jnp.concatenate(X0s); XL = jnp.concatenate(XLs)

    Tsched = jnp.zeros((STEPS,))
    xf, uf, rej = mini["run"](key, x0, u0, X0, XL, a, b, Tsched)

    # адиабатическое протаскивание из соседней ячейки (по θ), если задано
    drag_ok = False
    if drag_from is not None:
        xd, ud = drag_from
        # линейная интерполяция b от a(θ=0) к −a(θ=π) за DRAGSTEPS, затем доминимизация
        DRAG = 1500 if SMOKE else 2500
        a0, _ = M.apparatus_axes(False)
        for frac in np.linspace(0.0, 1.0, 6):
            bb = (1 - frac) * a0 + frac * b
            bb = bb / (jnp.linalg.norm(bb) + 1e-12)
            xd, ud, rd = mini["run"](jr.fold_in(key, int(frac * 100)), xd, ud,
                                     X0[:1], XL[:1], a, bb, jnp.zeros((DRAG // 6,)))
        sdr, tdr, degdr = M.classify_batch(ud, a, b)
        drag_ok = bool(((sdr == s) & (tdr == t) & ~degdr).any())

    # классификация и сектор (branch-канонические реперы с учётом θ)
    sarr, tarr, degen = M.classify_batch(uf, a, b)
    sa, sb = M.branch_frame_signs(s, t, theta_pi)
    UA, UB = M.frame_for_axis(sa), M.frame_for_axis(sb)
    par = np.array([float(I.parity(xf[i], uf[i], UA, UB)) for i in range(xf.shape[0])])
    sector_val = -1.0 if sector_odd else 1.0
    # сходимость (остаточный |ΔE|)
    E0 = np.asarray(jax.vmap(lambda x, u: M.e_meas(x, u, a, b, PARAMS))(x0, u0))
    Ef = np.asarray(jax.vmap(lambda x, u: M.e_meas(x, u, a, b, PARAMS))(xf, uf))

    in_branch = (np.asarray(sarr) == s) & (np.asarray(tarr) == t)
    no_wall = np.asarray(rej) == 0
    sect_ok = par == sector_val
    non_degen = ~np.asarray(degen)
    good = in_branch & no_wall & sect_ok & non_degen

    n_good = int(good.sum())
    n_degen = int(np.asarray(degen).sum())
    if n_good > 0 or drag_ok:
        verdict = "EXISTS"
    elif n_degen > NSTARTS // 2:
        verdict = "DEGENERATE"
    else:
        verdict = "NOT-FOUND"
    return dict(verdict=verdict, n_good=n_good, n_degen=n_degen,
                drag_ok=drag_ok, E_min=float(Ef.min()),
                rej_total=float(np.asarray(rej).sum()),
                branch_reached=int(in_branch.sum()),
                sector_ok=int(sect_ok.sum()),
                x_min=xf[int(np.argmin(Ef))], u_min=uf[int(np.argmin(Ef))])


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260713)
    print(f"[census {tag}] N={N} starts={NSTARTS} steps={STEPS}")

    cells = list(itertools.product([False, True], [1, -1], [1, -1], [False, True]))
    table = {}
    # сначала чётные ячейки (эталон разрешённых) — используем их минимумы для drag
    even_minima = {}
    for sector_odd, s, t, theta_pi in cells:
        key, sk = jr.split(key)
        drag = None
        if theta_pi and (sector_odd, s, t, False) in even_minima:  # drag из θ=0 той же ветви/сектора
            xm, um = even_minima[(sector_odd, s, t, False)]
            drag = (xm[None], um[None])
        r = run_cell(sector_odd, s, t, theta_pi, sk, drag_from=drag)
        even_minima[(sector_odd, s, t, theta_pi)] = (r.pop("x_min"), r.pop("u_min"))
        cellname = f"{'odd' if sector_odd else 'even'}|s{'+' if s>0 else '-'}t{'+' if t>0 else '-'}|θ{'π' if theta_pi else '0'}"
        table[cellname] = r
        print(f"  {cellname:22s}: {r['verdict']:10s} good={r['n_good']:2d}/{NSTARTS} "
              f"degen={r['n_degen']:2d} drag={int(r['drag_ok'])} E_min={r['E_min']:.3f} rej={r['rej_total']:.0f}")

    # ИТОГ по D-H3a: нечёт, выровнено (s=t), θ=0
    aligned_odd_0 = [table[f"odd|s{sg}t{sg}|θ0"] for sg in ["+", "-"]]
    all_exists = all(v["verdict"] == "EXISTS" for v in table.values())
    dh3a_verdict = ("NOT-CONFIRMED (EXISTS в выровненных нечёт θ=0)"
                    if all(v["verdict"] == "EXISTS" for v in aligned_odd_0)
                    else "aligned-odd-θ0: " + ",".join(v["verdict"] for v in aligned_odd_0))
    print(f"  ⇒ D-H3a: {dh3a_verdict}")
    print(f"  ⇒ все ячейки EXISTS: {all_exists}")

    with open(os.path.join(RES, f"census_{tag}.json"), "w") as f:
        json.dump(dict(tag=tag, N=N, nstarts=NSTARTS, steps=STEPS,
                       cells=table, all_exists=bool(all_exists),
                       dh3a=dh3a_verdict), f, indent=2)
    print(f"  сводка → {RES}/census_{tag}.json")


if __name__ == "__main__":
    main()
