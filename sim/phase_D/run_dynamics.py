"""D1-C. Динамическая статистика (см. D1_prereg.md §D-H3b/c, аддендум §6).

Ячейки {чёт,нечёт}×{θ=0,π}×N∈{16,48}, M реплик, T∈{low,mid,high}. Несмещённое
приготовление (бассейн решает ветвь). Наблюдаемые: совместные (s,t) по каждому
кандидату; DEGENERATE; rejected_singular/шаг; частота «запрещённой» ветви
(выровнено в нечёт θ=0) vs T и N; тест слепоты; зеркальные пары.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_dynamics.py [--smoke]
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
import measurement as M
import readout as R

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")

PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=2.0, ell=1.0)
M_REPL = 60 if SMOKE else 200
STEPS = 2000 if SMOKE else 4000
NS = [16] if SMOKE else [16, 48]
TEMPS = [0.02, 0.05, 0.10]
LR = 5e-3


def run_cell(N, sector_odd, theta_pi, T, key):
    """Одна ячейка: прогон M реплик, измерение (s,t) по всем кандидатам."""
    a, b = M.apparatus_axes(theta_pi)
    mini = M.build_minimizer(PARAMS, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, sector_odd, M_REPL, key)
    # релаксация при T, затем «доохлаждение» до T≈0 для чёткой ветви
    k1, k2 = jr.split(key)
    x, u, rej1 = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b,
                             jnp.full((STEPS,), T))
    x, u, rej2 = mini["run"](k2, x, u, p["X0"], p["XL"], a, b,
                             jnp.full((STEPS // 2,), 0.0))
    rej = np.asarray(rej1) + np.asarray(rej2)
    cand = R.eval_all_batch(x, u, a, b)
    # DEGENERATE по осевой резкости
    _, _, degen = M.classify_batch(u, a, b)
    return dict(cand=cand, degen=np.asarray(degen), rej=rej,
                n_steps=STEPS + STEPS // 2, x=x, u=u, a=a, b=b)


def joint_table(L, R_, degen):
    """Совместная таблица (s,t) → счётчики [++,+-,-+,--] + degen."""
    keep = ~degen
    Lk, Rk = L[keep], R_[keep]
    tab = {"++": int(((Lk > 0) & (Rk > 0)).sum()),
           "+-": int(((Lk > 0) & (Rk < 0)).sum()),
           "-+": int(((Lk < 0) & (Rk > 0)).sum()),
           "--": int(((Lk < 0) & (Rk < 0)).sum())}
    tab["degen"] = int(degen.sum())
    return tab


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260713 + 1)
    print(f"[dynamics {tag}] M={M_REPL} steps={STEPS} N={NS} T={TEMPS}")

    results = {}
    forbidden = []   # (N, T, freq aligned-odd-θ0 по AX)
    blind_accum = {name: [] for name in R.CANDIDATES}

    for N in NS:
        for sector_odd, theta_pi in itertools.product([False, True], [False, True]):
            for T in TEMPS:
                key, sk = jr.split(key)
                r = run_cell(N, sector_odd, theta_pi, T, sk)
                cellname = f"N{N}|{'odd' if sector_odd else 'even'}|θ{'π' if theta_pi else '0'}|T{T}"
                # таблицы по кандидатам
                tabs = {}
                for name in R.CANDIDATES:
                    L, R_ = r["cand"][name]
                    tabs[name] = joint_table(L, R_, r["degen"])
                rej_ps = float(r["rej"].sum()) / (M_REPL * r["n_steps"])
                results[cellname] = dict(tables=tabs, rej_per_step=rej_ps,
                                         degen=int(r["degen"].sum()))
                # частота «запрещённой» ветви (выровнено s=t) по AX в нечёт θ0
                if sector_odd and not theta_pi:
                    axt = tabs["AX"]
                    tot = axt["++"] + axt["+-"] + axt["-+"] + axt["--"]
                    aligned = axt["++"] + axt["--"]
                    freq = aligned / max(tot, 1)
                    forbidden.append(dict(N=N, T=T, aligned_freq=freq,
                                          aligned=aligned, total=tot))
                # тест слепоты на этой ячейке
                bl, _ = R.blindness_test(r["x"], r["u"], r["a"], r["b"])
                for name in R.CANDIDATES:
                    blind_accum[name].append(bl[name])
                print(f"  {cellname:26s}: AX={tabs['AX']}  rej/step={rej_ps:.2e}")

    # зеркальный контроль (обязателен): odd θ0 при T_mid, (a,b)→(−a,−b) + u→u⊗j.
    # AX-таблица должна отобразиться sign-флипом ветвей (++↔−−, +-↔-+).
    key, mk = jr.split(key)
    rc = run_cell(NS[0], True, False, TEMPS[1], mk)
    L, Rr = rc["cand"]["AX"]
    tab_orig = joint_table(L, Rr, rc["degen"])
    # зеркало: осевые оси −a,−b ⇒ (s,t) флипаются; проверяем на ТЕХ ЖЕ конфигурациях
    am, bm = -rc["a"], -rc["b"]
    Lm, Rm = jax.vmap(R.cand_AX, in_axes=(0, 0, None, None))(rc["x"], rc["u"], am, bm)
    tab_mir = joint_table(np.asarray(Lm), np.asarray(Rm), rc["degen"])
    mirror_ok = (tab_orig["++"] == tab_mir["--"] and tab_orig["--"] == tab_mir["++"] and
                 tab_orig["+-"] == tab_mir["-+"] and tab_orig["-+"] == tab_mir["+-"])
    print(f"  ЗЕРКАЛО (odd θ0): orig={tab_orig} mirror(−a,−b)={tab_mir} симметрично={mirror_ok}")

    # сводка слепоты (среднее по ячейкам)
    blind_mean = {name: float(np.mean(v)) for name, v in blind_accum.items()}
    print("  BLINDNESS (среднее agree_frac по ячейкам; 1.0=слеп к лифту):",
          {k: round(v, 3) for k, v in blind_mean.items()})

    # различие кандидатов с AX (совпадают ли таблицы во всех ячейках)
    differs_from_AX = {}
    for name in R.CANDIDATES:
        if name == "AX":
            continue
        diff_cells = [cn for cn, r in results.items()
                      if r["tables"][name] != r["tables"]["AX"]]
        differs_from_AX[name] = diff_cells
    print("  Кандидаты, чья таблица отличается от AX (в каких ячейках):")
    for name, cells in differs_from_AX.items():
        print(f"    {name:6s}: {len(cells)} ячеек " + (f"(напр. {cells[0]})" if cells else ""))

    print("  ЧАСТОТА выровненной ветви в нечёт θ0 (D-H3b), vs (N,T):")
    for f in forbidden:
        print(f"    N={f['N']} T={f['T']}: aligned/total={f['aligned']}/{f['total']} = {f['aligned_freq']:.3f}")

    summary = dict(tag=tag, M=M_REPL, NS=NS, TEMPS=TEMPS, cells=results,
                   blindness_mean=blind_mean,
                   differs_from_AX={k: len(v) for k, v in differs_from_AX.items()},
                   forbidden_aligned=forbidden,
                   mirror=dict(orig=tab_orig, mirror=tab_mir, symmetric=bool(mirror_ok)))
    with open(os.path.join(RES, f"dynamics_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  сводка → {RES}/dynamics_{tag}.json")


if __name__ == "__main__":
    main()
