"""Свип θ по ячейкам эксперимента, счётчики ветвей, сходимость, отчёт.

Поддержаны три формата конфига:
  - одиночная ячейка (R0): physics.k_e + dynamics.lr;
  - κ-сетка (R1): kappa_grid, k_e = κ·(N−1)·k_c, lr = 0.5/(k_e+k_c);
  - decay-сетка (R2): decay_grid, T0=1.0, k_e из лучшей ячейки R1.
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from . import analysis, plots
from .dynamics import branch_counts, build_relaxer, classify
from .frames import haar_quaternions


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setting_vectors(theta):
    """Оси зажимов: a = e_z (θ=0), b под физическим углом θ в плоскости x–z
    (a·b = cos θ, согласовано с ribbon_model_note/chord.py)."""
    a = jax.numpy.array([0.0, 0.0, 1.0], dtype=jax.numpy.float32)
    b = jax.numpy.array([np.sin(theta), 0.0, np.cos(theta)], dtype=jax.numpy.float32)
    return a, b


# --------------------------------------------------------------------------- #
#  Построение ячеек
# --------------------------------------------------------------------------- #
def build_cells(cfg, mode="full"):
    phys, dyn, run = cfg["physics"], cfg["dynamics"], cfg["run"]
    N = int(phys["N"])
    k_c = float(phys["k_c"])
    base = {
        "N": N,
        "k_c": k_c,
        "spinor": bool(phys.get("spinor", False)),
        "elastic": str(phys.get("elastic", "geodesic")),
        "T0": float(dyn["T0"]),
        "decay": float(dyn["decay"]),
        "steps": int(dyn["steps"]),
        "B": int(run["B"]),
        "seeds": list(run["seeds"]),
    }

    sw = cfg["sweep"]
    thetas_deg = np.arange(
        sw["theta_deg_start"], sw["theta_deg_stop"] + 1e-6, sw["theta_deg_step"]
    )
    if mode == "smoke":
        sm = cfg["smoke"]
        base["B"] = int(sm["B"])
        base["steps"] = int(sm.get("steps", base["steps"]))  # block-конфиг может не иметь steps
        base["seeds"] = list(sm["seeds"])
        thetas_deg = np.array(sm["theta_deg"], dtype=np.float64)

    cells = []
    if "elastic_grid" in cfg and "kappa_grid" in cfg:  # R5: 2×2 (elastic × κ)
        short = {"geodesic": "geo", "spinor": "spin", "chordal": "chord"}
        for el in cfg["elastic_grid"]:
            for kappa in cfg["kappa_grid"]:
                k_e = float(kappa) * (N - 1) * k_c
                cells.append({**base, "elastic": str(el), "k_e": k_e,
                              "lr": 0.5 / (k_e + k_c), "kappa": float(kappa),
                              "label": f"{short.get(el, el)} κ={kappa:g}"})
    elif "kappa_grid" in cfg:  # R1
        for kappa in cfg["kappa_grid"]:
            k_e = float(kappa) * (N - 1) * k_c
            cells.append({**base, "k_e": k_e, "lr": 0.5 / (k_e + k_c),
                          "kappa": float(kappa), "label": f"κ={kappa:g}"})
    elif "decay_grid" in cfg:  # R2 / R5b
        k_e = float(phys["k_e"])
        kappa = float(phys.get("kappa", k_e / ((N - 1) * k_c)))
        for d in cfg["decay_grid"]:
            cells.append({**base, "k_e": k_e, "lr": 0.5 / (k_e + k_c), "kappa": kappa,
                          "decay": float(d), "decay_val": float(d),
                          "label": f"decay={d:g}"})
    else:  # R0-стиль: одна ячейка
        k_e = float(phys["k_e"])
        lr = float(dyn.get("lr", 0.5 / (k_e + k_c)))
        cells.append({**base, "k_e": k_e, "lr": lr, "label": cfg["name"]})

    return cells, np.radians(thetas_deg), thetas_deg


# --------------------------------------------------------------------------- #
#  Прогон одной ячейки (со сходимостью)
# --------------------------------------------------------------------------- #
def run_cell(cell, thetas, conv_cfg=None, seed_override=None):
    """Прогон ячейки с критерием сходимости и однократным удвоением steps.

    Критерий (SPEC/архитектор): доля лент с |ΔE|/шаг > de_threshold < frac_threshold.
    """
    de_thr = float(conv_cfg["de_threshold"]) if conv_cfg else 0.0
    frac_thr = float(conv_cfg["frac_threshold"]) if conv_cfg else 1.0
    max_doublings = int(conv_cfg["max_doublings"]) if conv_cfg else 0

    t0 = time.time()
    doublings = 0
    steps_history = []
    while True:
        relaxer = build_relaxer(cell)
        runner, probe = relaxer["run"], relaxer["probe"]
        N, B = cell["N"], cell["B"]
        seeds = [int(seed_override)] if seed_override is not None else cell["seeds"]

        counts = np.zeros((len(seeds), len(thetas), 4), dtype=np.int64)
        e_final = np.zeros((len(seeds), len(thetas)), dtype=np.float64)
        conv_frac = np.zeros((len(seeds), len(thetas)), dtype=np.float64)

        for si, seed in enumerate(seeds):
            base_key = jax.random.PRNGKey(int(seed))
            for ti, theta in enumerate(thetas):
                a, b = setting_vectors(theta)
                tk = jax.random.fold_in(base_key, ti)
                k_init, k_noise = jax.random.split(tk)
                q0 = haar_quaternions(k_init, (B, N))
                qf, e_trace = runner(k_noise, q0, a, b)
                s, t = classify(qf, a, b)
                cnt = branch_counts(s, t)
                deltas = probe(qf, a, b)
                cnt.block_until_ready()
                counts[si, ti] = np.asarray(cnt)
                e_final[si, ti] = float(e_trace[-1])
                conv_frac[si, ti] = float(np.mean(np.asarray(deltas) > de_thr))

        steps_history.append(cell["steps"])
        max_frac = float(conv_frac.max())
        converged = max_frac < frac_thr
        if converged or doublings >= max_doublings:
            break
        # однократное удвоение steps
        cell = {**cell, "steps": cell["steps"] * 2}
        doublings += 1

    return {
        "cell": cell,
        "counts": counts,
        "e_final": e_final,
        "conv_frac": conv_frac,
        "max_conv_frac": max_frac,
        "converged": converged,
        "steps_used": cell["steps"],
        "steps_history": steps_history,
        "doublings": doublings,
        "seeds": seeds,
        "elapsed_s": time.time() - t0,
    }


def run_cell_blocks(cell, thetas, block_cfg, seed_override=None):
    """Блочный протокол сходимости по НАБЛЮДАЕМОЙ (ответ на ARCH-Q#1, R5).

    Каждую (сид, θ) релаксируем блоками block_steps, продолжая с прошлого q, пока
    |ΔE| между блоками < max(sigma_mult·σ_MC, e_floor) или не достигнут ceiling.
    Плюс диагностика кинков (плотность и разбивка по ветвям).
    """
    from .dynamics import holonomy, kink_count

    block = int(block_cfg["block_steps"])
    ceiling = int(block_cfg["ceiling"])
    smult = float(block_cfg.get("sigma_mult", 2.0))
    e_floor = float(block_cfg.get("e_floor", 0.01))
    cold_T = float(block_cfg.get("cold_T", 1e-4))  # проверять сходимость только при T<cold_T
    T0, decay = cell["T0"], cell["decay"]
    N, B = cell["N"], cell["B"]
    seeds = [int(seed_override)] if seed_override is not None else cell["seeds"]

    relaxer = build_relaxer({**cell, "steps": block})
    runner = relaxer["run"]

    def npbc(s, t):  # branch counts на numpy-подмножествах [pp,pm,mp,mm]
        return np.array([np.sum((s > 0) & (t > 0)), np.sum((s > 0) & (t < 0)),
                         np.sum((s < 0) & (t > 0)), np.sum((s < 0) & (t < 0))])

    nS, nT = len(seeds), len(thetas)
    counts = np.zeros((nS, nT, 4), dtype=np.int64)
    counts_M2 = np.zeros((nS, nT, 4), dtype=np.int64)          # t̃ = h·sign(n_B·b)
    counts_hplus = np.zeros((nS, nT, 4), dtype=np.int64)       # ветви при h=+1
    counts_hminus = np.zeros((nS, nT, 4), dtype=np.int64)      # ветви при h=−1
    n_hminus = np.zeros((nS, nT), dtype=np.int64)
    kink_sum_by_branch = np.zeros((nS, nT, 4), dtype=np.float64)
    kink_density = np.zeros((nS, nT), dtype=np.float64)
    steps_per = np.zeros((nS, nT), dtype=np.int64)
    conv_flag = np.zeros((nS, nT), dtype=bool)

    t0 = time.time()
    for si, seed in enumerate(seeds):
        base_key = jax.random.PRNGKey(int(seed))
        for ti, theta in enumerate(thetas):
            a, b = setting_vectors(theta)
            tk = jax.random.fold_in(base_key, ti)
            k_init, k_noise = jax.random.split(tk)
            q = haar_quaternions(k_init, (B, N))
            e_prev, total, converged = None, 0, False
            while total < ceiling:
                k_noise, sub = jax.random.split(k_noise)
                q, _ = runner(sub, q, a, b, jnp.int32(total))  # step0=total для отжига
                total += block
                # проверять сходимость только когда холодно (T<cold_T)
                T_now = T0 * decay ** total
                if T0 > 0 and T_now >= cold_T:
                    e_prev = None
                    continue
                s, t = classify(q, a, b)
                e = float(np.mean(np.asarray(s) * np.asarray(t)))
                sigma = np.sqrt(max(1.0 - e * e, 0.0) / B)
                if e_prev is not None and abs(e - e_prev) < max(smult * sigma, e_floor):
                    converged = True
                    break
                e_prev = e
            # финальные наблюдаемые
            s = np.asarray(classify(q, a, b)[0]); t = np.asarray(classify(q, a, b)[1])
            h = np.asarray(holonomy(q))
            kc = np.asarray(kink_count(q))
            counts[si, ti] = npbc(s, t)
            counts_M2[si, ti] = npbc(s, h * t)               # M2: голономно-одетая t̃
            counts_hplus[si, ti] = npbc(s[h > 0], t[h > 0])
            counts_hminus[si, ti] = npbc(s[h < 0], t[h < 0])
            n_hminus[si, ti] = int(np.sum(h < 0))
            bidx = np.where(s > 0, np.where(t > 0, 0, 1), np.where(t > 0, 2, 3))
            for k in range(4):
                kink_sum_by_branch[si, ti, k] = kc[bidx == k].sum()
            kink_density[si, ti] = kc.mean() / (N - 1)
            steps_per[si, ti] = total
            conv_flag[si, ti] = converged

    return {
        "cell": cell,
        "counts": counts,
        "counts_M2": counts_M2,
        "counts_hplus": counts_hplus,
        "counts_hminus": counts_hminus,
        "n_hminus": n_hminus,
        "kink_sum_by_branch": kink_sum_by_branch,
        "kink_density": kink_density,
        "steps_per": steps_per,
        "conv_flag": conv_flag,
        "converged": bool(conv_flag.all()),
        "frac_converged": float(conv_flag.mean()),
        "max_steps": int(steps_per.max()),
        "steps_used": int(steps_per.max()),
        "doublings": 0,
        "seeds": seeds,
        "block": True,
        "elapsed_s": time.time() - t0,
    }


def run_experiment(cfg, mode="full", seed_override=None):
    cells, thetas, thetas_deg = build_cells(cfg, mode)
    block_cfg = cfg["dynamics"].get("block_convergence")
    if mode == "smoke" and block_cfg and "smoke" in cfg:
        for k in ("block_steps", "ceiling"):
            if k in cfg["smoke"]:
                block_cfg = {**block_cfg, k: cfg["smoke"][k]}
    conv_cfg = cfg["dynamics"].get("convergence")
    results = []
    for cell in cells:
        print(f"  [{cell['label']}] elastic={cell['elastic']} k_e={cell['k_e']:.3f} "
              f"lr={cell['lr']:.2e} T0={cell['T0']}")
        if block_cfg:
            res = run_cell_blocks(cell, thetas, block_cfg, seed_override=seed_override)
            print(f"    → {res['elapsed_s']:.1f} c; сошлось {res['frac_converged']*100:.0f}% "
                  f"(θ,сид); max_steps={res['max_steps']}")
        else:
            res = run_cell(cell, thetas, conv_cfg=conv_cfg, seed_override=seed_override)
            print(f"    → {res['elapsed_s']:.1f} c; сходимость max_frac={res['max_conv_frac']:.4f} "
                  f"({'ok' if res['converged'] else 'NOT converged'}); steps={res['steps_used']}"
                  + (f" (удвоено ×{res['doublings']})" if res['doublings'] else ""))
        results.append(res)
    return {"name": cfg["name"], "mode": mode, "thetas": thetas,
            "thetas_deg": thetas_deg, "cells": results, "cfg": cfg}


# --------------------------------------------------------------------------- #
#  Отчёт
# --------------------------------------------------------------------------- #
def _gpu_mem_mb():
    try:
        stats = jax.devices()[0].memory_stats()
        peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use", 0)
        return peak / 1e6
    except Exception:
        return None


def _analyse_cell(res, thetas):
    counts = res["counts"]
    counts_sum = counts.sum(0)
    E = analysis.E_from_counts(counts_sum)             # физический наблюдаемый (ферро)
    ctrl = analysis.check_controls(counts_sum, thetas)
    repro = analysis.reproducibility(counts[0], counts[1]) if counts.shape[0] >= 2 else None
    # фиты семейств SPEC — в синглетной конвенции (глобальный флип t̃=−t)
    cs = analysis.singlet_counts(counts_sum)
    cmp = analysis.compare_models(cs, thetas)
    boot = analysis.bootstrap_p(cs, thetas, n_boot=1000, seed=0)
    harm = analysis.harmonics(thetas, E)

    aics = {"пила": cmp["aic_saw"], "хорда-p": cmp["aic_chord_p"],
            "хорда-p=2": cmp["aic_chord_p2"]}
    best = min(aics, key=aics.get)
    amp = float(np.max(np.abs(E)))
    i0 = int(np.argmin(np.abs(thetas)))
    ferro = bool(E[i0] > 0)  # знак корреляции при θ→0: ферро E(0)>0
    return {"E": E, "ctrl": ctrl, "repro": repro, "cmp": cmp, "boot": boot,
            "harm": harm, "aics": aics, "best": best, "amp": amp,
            "ferro": ferro, "E0": float(E[i0]), "counts_sum": counts_sum}


def _slug(label):
    return (label.replace("κ", "k").replace("=", "").replace(".", "p").replace(" ", "_"))


def write_report(exp, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = exp["cfg"]
    thetas = exp["thetas"]
    deg = exp["thetas_deg"]

    # Анализ каждой ячейки + сырьё на диск.
    analyses = []
    npz_payload = {"thetas": thetas, "thetas_deg": deg}
    for res in exp["cells"]:
        an = _analyse_cell(res, thetas)
        analyses.append(an)
        lbl = _slug(res["cell"]["label"])
        npz_payload[f"counts_{lbl}"] = res["counts"]
        # график E(θ) на ячейку
        seeds_E = [analysis.E_from_counts(res["counts"][si]) for si in range(res["counts"].shape[0])]
        plots.plot_E_curve(thetas, an["E"], outdir / f"E_{lbl}_{exp['mode']}.png",
                           seeds_E=seeds_E, title=f"{exp['name']} {res['cell']['label']} ({exp['mode']})")
    np.savez(outdir / f"counts_{exp['mode']}.npz", **npz_payload)

    # Сводный график всех ячеек.
    _plot_all_cells(exp, analyses, outdir / f"E_all_{exp['mode']}.png")

    is_r5b = "decay_grid" in cfg and "block_convergence" in cfg["dynamics"]
    if "elastic_grid" in cfg:
        md = _render_r5(exp, analyses, deg)
    elif is_r5b:
        md = _render_r5b(exp, analyses, deg, outdir)
    else:
        md = _render(exp, analyses, deg, outdir)
    (outdir / "report.md").write_text(md, encoding="utf-8")

    total_elapsed = sum(r["elapsed_s"] for r in exp["cells"])
    return {"path": str(outdir / "report.md"), "analyses": analyses,
            "total_elapsed": total_elapsed}


def _plot_all_cells(exp, analyses, out_path):
    import matplotlib.pyplot as plt

    deg = exp["thetas_deg"]
    grid = np.linspace(0, np.pi, 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.degrees(grid), -np.cos(grid), "k--", lw=1, alpha=0.5, label="−cos θ (p=2)")
    ax.plot(np.degrees(grid), 2 * grid / np.pi - 1, "b:", lw=1, alpha=0.5, label="пила")
    ax.axhline(0, color="gray", lw=0.5)
    for res, an in zip(exp["cells"], analyses):
        ax.plot(deg, an["E"], "o-", ms=4, label=res["cell"]["label"])
    ax.set_xlabel("θ, градусы"); ax.set_ylabel("E(θ) = ⟨s·t⟩")
    ax.set_title(f"{exp['name']} ({exp['mode']}): E(θ) по ячейкам")
    ax.set_ylim(-1.1, 1.1); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _render(exp, analyses, deg, outdir):
    cfg = exp["cfg"]
    L = []
    A = L.append
    A(f"# {exp['name']} — отчёт ({exp['mode']})\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрированное предсказание:** {cfg.get('prediction', '')}\n")

    c0 = exp["cells"][0]["cell"]
    total = sum(r["elapsed_s"] for r in exp["cells"])
    mem = _gpu_mem_mb()
    A("## Общая конфигурация\n")
    A("```")
    A(f"N={c0['N']}  k_c={c0['k_c']}  spinor={c0['spinor']}  elastic={c0['elastic']}")
    A(f"B={c0['B']}  seeds={c0['seeds']}  θ (°): {np.round(deg, 2).tolist()}")
    A("```")
    A(f"\nВремя: **{total:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else "") + "\n")

    # Баннер сходимости: если хоть одна ячейка не сошлась — результаты суть транзиенты.
    if any(not r["converged"] for r in exp["cells"]):
        A("> ## ⚠️ СТАТУС: НЕ СОШЛОСЬ — РЕЗУЛЬТАТЫ СУТЬ ТРАНЗИЕНТЫ\n>\n"
          "> Ни одна/не все ячейки не прошли критерий сходимости даже после удвоения "
          "steps. Протокол паранойи (SPEC §9.2) показал, что E(θ) продолжает дрейфовать "
          "с числом шагов: приведённые ниже кривые — снимки НЕПОЛНОЙ релаксации, а НЕ "
          "равновесие. Истинное равновесие см. `paranoia.md` и `convergence_study.md`:\n>\n"
          "> - κ=1 сходится (~30k шагов) к неполной ферро-корреляции E≈0.38·cosθ "
          "(не насыщается до ±1 — вне семейства хорды);\n"
          "> - κ=10 сходится (>100k шагов) к ферро-СТУПЕНИ при 90° (p→∞, PR-box);\n"
          "> - никакого p=2; лента ВЫРАВНИВАЕТ концы (ферро), это не синглет.\n>\n"
          "> Любой показатель p̂ в таблице ниже относится к ТРАНЗИЕНТУ и не является "
          "физическим результатом. R2 не запускался: наука на несошедшемся базисе "
          "запрещена (CLAUDE.md).\n")

    A(f"![E(θ) по ячейкам](E_all_{exp['mode']}.png)\n")

    # Сводная таблица.
    A("## Сводка по ячейкам (SPEC §4)\n")
    is_r2 = "decay_grid" in cfg
    key_col = "decay (T_fin)" if is_r2 else "κ"
    A(f"| {key_col} | k_e | steps | амплитуда max\\|E\\| | знак | лучшая модель (ΔAIC) | p̂ ± 95% CI | A1(cosθ) | A3(cos3θ) | сход. |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for res, an in zip(exp["cells"], analyses):
        cell = res["cell"]
        aics_sorted = sorted(an["aics"].items(), key=lambda kv: kv[1])
        d_aic = aics_sorted[1][1] - aics_sorted[0][1]
        lo, hi = an["boot"]["ci95"]
        if is_r2:
            T_fin = cell["T0"] * cell["decay"] ** cell["steps"]
            key = f"{cell['decay_val']:g} ({T_fin:.1e})"
        else:
            key = f"{cell['kappa']:g}"
        conv = "✅" if res["converged"] else f"⚠️×{res['doublings']}"
        sgn = "ферро (+)" if an["ferro"] else "антиферро (−)"
        A(f"| {key} | {cell['k_e']:.2f} | {res['steps_used']} | {an['amp']:.3f} | {sgn} "
          f"| {an['best']} (Δ{d_aic:.0f}) | {an['cmp']['p_hat']:.3f} "
          f"[{lo:.3f}, {hi:.3f}] | {an['harm']['A1']:+.3f} | {an['harm']['A3']:+.3f} | {conv} |")
    A("")
    A("*Знак: ферро (+) = E(0)=+1, концы выравниваются; антиферро (−) = E(0)=−1, "
      "синглетная конвенция закона хорды. p̂ — показатель ФОРМЫ семейства хорды "
      "в найденном знаке.*\n")

    # Детали каждой ячейки.
    for res, an in zip(exp["cells"], analyses):
        _render_cell(A, res, an, deg, exp["mode"])

    _render_verdict(A, exp, analyses, is_r2)
    return "\n".join(L) + "\n"


def _render_cell(A, res, an, deg, mode):
    cell = res["cell"]
    lbl = cell["label"]
    lbl_f = lbl.replace("=", "").replace("κ", "kappa").replace(".", "p")
    A(f"## Ячейка {lbl}\n")
    A(f"lr={cell['lr']:.3e}  steps={res['steps_used']}"
      + (f" (удвоено ×{res['doublings']}, история {res['steps_history']})" if res["doublings"] else "")
      + f"  T0={cell['T0']}  decay={cell['decay']}  время {res['elapsed_s']:.1f} с\n")
    A(f"![E]({'E_' + lbl_f + '_' + mode + '.png'})\n")

    ctrl, repro = an["ctrl"], an["repro"]
    A("**Контроли §4.2:** "
      f"маргиналы {'✅' if ctrl['all_marg_ok'] else '❌'}; "
      f"±-симметрия {'✅' if ctrl['all_sym_ok'] else '❌'} "
      f"(max {max(ctrl['sym_same_sigma'].max(), ctrl['sym_opp_sigma'].max()):.2f}σ); ")
    if repro is not None:
        A(f"воспроизводимость (max-статистика) {'✅' if repro['passes'] else '❌'} "
          f"max|z|={repro['max_z']:.2f} vs порог {repro['z_thresh']:.2f} "
          f"(семейный p={repro['global_p']:.3f}); ")
    A(f"сходимость: доля лент |ΔE|/шаг>1e-6 = {res['max_conv_frac']:.4f} "
      f"({'сошлось' if res['converged'] else 'НЕ сошлось'}).\n")

    cmp = an["cmp"]
    sgn = "ферро E(0)=+1" if an["ferro"] else "антиферро E(0)=−1"
    A(f"**Знак корреляции:** {sgn}. "
      f"**Фиты в этом знаке (AIC, меньше — лучше):** пила={cmp['aic_saw']:.0f}, "
      f"хорда-p={cmp['aic_chord_p']:.0f} (p̂={cmp['p_hat']:.3f}), "
      f"хорда-p=2={cmp['aic_chord_p2']:.0f} → лучшая: **{an['best']}**. "
      f"Bootstrap p = {an['boot']['p_mean']:.3f}, 95% CI [{an['boot']['ci95'][0]:.3f}, {an['boot']['ci95'][1]:.3f}].\n")

    # таблица E(θ)
    counts_sum = an["counts_sum"]
    n = counts_sum.sum(-1)
    A("<details><summary>Таблица E(θ), маргиналы, симметрия</summary>\n")
    A("| θ° | n | E(θ) | P(s=+) | P(t=+) | sym_same(σ) | sym_opp(σ) |")
    A("|---|---|---|---|---|---|---|")
    for i, th in enumerate(deg):
        A(f"| {th:.1f} | {int(n[i])} | {an['E'][i]:+.4f} | {ctrl['p_s'][i]:.4f} "
          f"| {ctrl['p_t'][i]:.4f} | {ctrl['sym_same_sigma'][i]:.2f} | {ctrl['sym_opp_sigma'][i]:.2f} |")
    A("\n</details>\n")


def _render_verdict(A, exp, analyses, is_r2):
    cfg = exp["cfg"]
    A("## Вердикт против пре-регистрации\n")
    A(f"Предсказание: {cfg.get('prediction', '')}\n")
    if is_r2:
        amp_str = ", ".join(
            f"{res['cell']['decay_val']:g}:{an['amp']:.3f}"
            for res, an in zip(exp["cells"], analyses))
        best_str = ", ".join(
            f"{res['cell']['decay_val']:g}:{an['best']}"
            for res, an in zip(exp["cells"], analyses))
        A(f"- Амплитуды max|E| по скоростям отжига: {amp_str}.")
        A(f"- Лучшие модели: {best_str}.")
        A("- Проверка гипотезы «термическая мера сглаживает кривую, форма семейства "
          "не меняется» — сопоставить амплитуды и лучшие модели с R1 (см. сводку).")
    else:
        for res, an in zip(exp["cells"], analyses):
            sgn = "ферро" if an["ferro"] else "антиферро"
            p2_flag = "← p̂≈2 (!)" if abs(an["cmp"]["p_hat"] - 2.0) < 0.15 else ""
            A(f"- **{res['cell']['label']}**: амплитуда {an['amp']:.3f}, {sgn}, лучшая модель "
              f"{an['best']}, p̂={an['cmp']['p_hat']:.3f} {p2_flag}")
        anyp2 = any(abs(an["cmp"]["p_hat"] - 2.0) < 0.15 for an in analyses)
        anyferro = any(an["ferro"] for an in analyses)
        if anyferro:
            A("\n> ⚠️ Знак корреляции ФЕРРО (E(0)=+1): изотропная упругая лента "
              "ВЫРАВНИВАЕТ концы, а не антивыравнивает (в отличие от синглета E=−cosθ). "
              "Пре-регистрированный фит хорды (антиферро) выродился на границу сетки — "
              "здесь показан фит формы с перебором знака.")
        if anyp2:
            A("\n> ⚠️ Где-то p̂≈2 ВОПРЕКИ пре-регистрации (ожидались пила/p≈1). "
              "Протокол паранойи SPEC §9.2 (второй сид ✓, удвоение N и B, контроли, "
              "поиск бага) — см. `paranoia.md`. Интерпретация — с архитектором (CLAUDE.md).")
        else:
            A("\n> p=2 не обнаружено ни в одной ячейке.")


# --------------------------------------------------------------------------- #
#  Отчёт R5 (geodesic vs spinor, кинки)
# --------------------------------------------------------------------------- #
def _kink_stats(res, N):
    """Плотность кинков и превышение «миноритарная − мажоритарная ветвь» (R5)."""
    counts_sum = res["counts"].sum(0).astype(float)          # (n_theta, 4)
    ksum = res["kink_sum_by_branch"].sum(0)                   # (n_theta, 4)
    mean_kink_branch = ksum / np.where(counts_sum > 0, counts_sum, 1.0)
    density_by_theta = res["kink_density"].mean(0)            # доля кинков/связь
    excess = []
    for ti in range(counts_sum.shape[0]):
        c = counts_sum[ti]
        if c.min() <= 0:
            continue
        mn, mj = int(np.argmin(c)), int(np.argmax(c))
        excess.append(mean_kink_branch[ti, mn] - mean_kink_branch[ti, mj])
    return {
        "density_mean": float(density_by_theta.mean()),
        "density_by_theta": density_by_theta,
        "minority_excess": float(np.mean(excess)) if excess else float("nan"),
    }


def _render_r5(exp, analyses, deg):
    cfg = exp["cfg"]
    cells = exp["cells"]
    N = cells[0]["cell"]["N"]
    kinks = [_kink_stats(res, N) for res in cells]
    # индексация по (elastic, κ)
    by_key = {(res["cell"]["elastic"], res["cell"]["kappa"]): (res, an, kk)
              for res, an, kk in zip(cells, analyses, kinks)}
    i0 = int(np.argmin(np.abs(deg)))  # индекс θ=0

    L = []; A = L.append
    A(f"# {exp['name']} — отчёт ({exp['mode']}): спинорная лента vs geodesic\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрация:** {cfg.get('prediction', '')}\n")

    c0 = cells[0]["cell"]
    total = sum(r["elapsed_s"] for r in cells)
    mem = _gpu_mem_mb()
    A("## Конфигурация\n```")
    A(f"N={c0['N']}  k_c={c0['k_c']}  B={c0['B']}  seeds={c0['seeds']}  T0={c0['T0']}")
    A(f"блочная сходимость: block={cfg['dynamics']['block_convergence']['block_steps']}, "
      f"ceiling={cfg['dynamics']['block_convergence']['ceiling']}, "
      f"крит. max_θ|ΔE|<max(2σ_MC,0.01)")
    A(f"θ (°): {np.round(deg, 2).tolist()}")
    A("```")
    A(f"\nВремя: **{total:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else "") + "\n")
    A(f"![E(θ) по ячейкам](E_all_{exp['mode']}.png)\n")

    # --- главная сводка geodesic vs spinor ---
    A("## Сводка: geodesic vs spinor (κ=1, κ=10)\n")
    A("| ячейка | κ | амп max\\|E\\| | E(0) | знак | A1 | A3 | A3/A1 | плотн. кинков | сошлось% | max_steps |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for res, an, kk in zip(cells, analyses, kinks):
        cell = res["cell"]
        h = an["harm"]
        a3a1 = h["A3"] / h["A1"] if abs(h["A1"]) > 1e-6 else float("nan")
        sgn = "ферро" if an["ferro"] else "антиферро"
        A(f"| {cell['label']} | {cell['kappa']:g} | {an['amp']:.3f} | {an['E'][i0]:+.3f} "
          f"| {sgn} | {h['A1']:+.3f} | {h['A3']:+.3f} | {a3a1:+.3f} "
          f"| {kk['density_mean']:.3f} | {res['frac_converged']*100:.0f} | {res['max_steps']} |")
    A("")

    # --- кинки ↔ ветвь ---
    A("## Диагностика кинков ↔ ветвь исхода\n")
    A("Гипотеза (пре-рег b): ленты в МИНОРИТАРНОЙ ветви несут больше кинков. "
      "Превышение = ⟨кинков|минор.ветвь⟩ − ⟨кинков|мажор.ветвь⟩ (на ленту, из "
      f"{N-1} связей), усреднено по θ.\n")
    A("| ячейка | плотн. кинков (доля связей) | превышение минор.−мажор. |")
    A("|---|---|---|")
    for res, kk in zip(cells, kinks):
        A(f"| {res['cell']['label']} | {kk['density_mean']:.4f} | {kk['minority_excess']:+.3f} |")
    A("")

    # --- контроли §4.2 ---
    A("## Контроли §4.2 по ячейкам\n")
    A("| ячейка | маргиналы | ±-симм (maxσ) | воспроизв. (max\\|z\\| vs порог) |")
    A("|---|---|---|---|")
    for res, an in zip(cells, analyses):
        ctrl, repro = an["ctrl"], an["repro"]
        rr = (f"{repro['max_z']:.2f} vs {repro['z_thresh']:.2f} "
              f"{'✅' if repro['passes'] else '❌'}") if repro else "—"
        A(f"| {res['cell']['label']} | {'✅' if ctrl['all_marg_ok'] else '❌'} "
          f"| {'✅' if ctrl['all_sym_ok'] else '❌'} "
          f"({max(ctrl['sym_same_sigma'].max(), ctrl['sym_opp_sigma'].max()):.2f}) | {rr} |")
    A("")

    # --- вердикт по пунктам пре-регистрации ---
    A("## Вердикт по пре-регистрации\n")
    all_ferro = all(an["ferro"] for an in analyses)
    A(f"**(a) знак ферро** (n(−q)=n(q), зажимы знако-слепы): "
      f"{'✅ подтверждено' if all_ferro else '❌ где-то антиферро'} "
      f"(все ячейки {'ферро' if all_ferro else 'разное'}).")
    spin_dens = [kk["density_mean"] for res, kk in zip(cells, kinks)
                 if res["cell"]["elastic"] == "spinor"]
    b_ok = all(d > 0 for d in spin_dens) and len(spin_dens) > 0
    A(f"\n**(b) кинки застревают при T=0, плотность>0**: "
      f"{'✅' if b_ok else '❌'} (spinor плотности: "
      f"{', '.join(f'{d:.3f}' for d in spin_dens)}).")
    # (c) амплитуда E(0) κ=1 spinor < geodesic — сравниваем с MC-шумом
    try:
        geo_an, geo_res = by_key[("geodesic", 1.0)][1], by_key[("geodesic", 1.0)][0]
        spin_an = by_key[("spinor", 1.0)][1]
        e0_geo, e0_spin = float(geo_an["E"][i0]), float(spin_an["E"][i0])
        amp_geo, amp_spin = geo_an["amp"], spin_an["amp"]
        n_tot = geo_res["counts"].sum()  # для σ_MC при θ=0
        sig = np.sqrt(max(1 - e0_geo ** 2, 0.0) / (geo_res["counts"][:, i0].sum())) * np.sqrt(2)
        dsig = abs(e0_geo - e0_spin) / sig if sig > 0 else 0.0
        A(f"\n**(c) амплитуда κ=1: spinor < geodesic**: по max\\|E\\| равны "
          f"({amp_geo:.3f} vs {amp_spin:.3f}); по E(0) spin {'ниже' if e0_spin < e0_geo else 'не ниже'} "
          f"({e0_geo:.3f} vs {e0_spin:.3f}, {dsig:.1f}σ). "
          f"Механизм (ловушки кинков) зависит от пункта (b).")
    except KeyError:
        A("\n**(c)** не хватает пары κ=1 geo/spin для сравнения.")
    # (d) A3/A1 — открытый вопрос
    A("\n**(d) форма A3/A1** — без прогноза, открытый вопрос. Значения:")
    for res, an in zip(cells, analyses):
        h = an["harm"]
        a3a1 = h["A3"] / h["A1"] if abs(h["A1"]) > 1e-6 else float("nan")
        A(f"  - {res['cell']['label']}: A3/A1 = {a3a1:+.3f} (A1={h['A1']:+.3f}, A3={h['A3']:+.3f})")
    A("\n> Интерпретация — с архитектором (CLAUDE.md). Сходимость: блочный критерий "
      "по наблюдаемой; ячейки, не достигшие 100%, ограничены потолком ceiling.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
#  Отчёт R5b (термический спинор + голономия + M2)
# --------------------------------------------------------------------------- #
def _E_safe(counts_2d):
    """E(θ) с nan там, где ветвей нет (для условных по h подмножеств)."""
    counts_2d = np.asarray(counts_2d, dtype=np.float64)
    n = counts_2d.sum(-1)
    E = (counts_2d * analysis.ST_SIGN).sum(-1) / np.where(n > 0, n, np.nan)
    return E


def _render_r5b(exp, analyses, deg, outdir):
    import matplotlib.pyplot as plt

    cfg = exp["cfg"]
    cells = exp["cells"]
    thetas = exp["thetas"]
    i0 = int(np.argmin(np.abs(deg)))
    L = []; A = L.append
    A(f"# {exp['name']} — отчёт ({exp['mode']}): термический спинор + голономия\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрация:** {cfg.get('prediction', '')}\n")
    A("**M2** — отдельный пре-регистрированный ВАРИАНТ МОДЕЛИ: голономно-одетая "
      "наблюдаемая t̃ = h·sign(n_B·b). Меняет ОПРЕДЕЛЕНИЕ наблюдаемой — сравнивать с M1 "
      "бок о бок, не смешивать.\n")

    c0 = cells[0]["cell"]
    total = sum(r["elapsed_s"] for r in cells)
    mem = _gpu_mem_mb()
    A("## Конфигурация\n```")
    A(f"spinor  κ={c0.get('kappa','?')}  N={c0['N']}  k_e={c0['k_e']:.1f}  B={c0['B']}  "
      f"seeds={c0['seeds']}  T0={c0['T0']}  lr={c0['lr']:.2e}")
    A(f"блочная сходимость после T<{cfg['dynamics']['block_convergence'].get('cold_T',1e-4)}, "
      f"ceiling={cfg['dynamics']['block_convergence']['ceiling']}")
    A("```")
    A(f"\nВремя: **{total:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else "") + "\n")
    A(f"![E(θ) M1 по ячейкам](E_all_{exp['mode']}.png)\n")

    # --- сводка по скоростям отжига ---
    A("## Сводка: отжиг → кинки, голономия, наблюдаемые M1/M2\n")
    A("| decay | T_fin | плотн. кинков | ⟨P(h=−1)⟩ | E_M1(0) | E_M2(0) | max_θ\\|E(h+)−E(h−)\\| | сошлось% |")
    A("|---|---|---|---|---|---|---|---|")
    stats = []
    for res, an in zip(cells, analyses):
        cell = res["cell"]
        B, ns = cell["B"], len(res["seeds"])
        ntot = B * ns
        E_m1 = an["E"]
        E_m2 = analysis.E_from_counts(res["counts_M2"].sum(0))
        p_hm = res["n_hminus"].sum(0) / ntot
        E_hp = _E_safe(res["counts_hplus"].sum(0))
        E_hm = _E_safe(res["counts_hminus"].sum(0))
        cond_gap = np.nanmax(np.abs(E_hp - E_hm)) if np.isfinite(E_hp - E_hm).any() else float("nan")
        kdens = float(res["kink_density"].mean())
        T_fin = cell["T0"] * cell["decay"] ** res["max_steps"]
        stats.append({"E_m1": E_m1, "E_m2": E_m2, "p_hm": p_hm, "E_hp": E_hp,
                      "E_hm": E_hm, "kdens": kdens, "cond_gap": cond_gap})
        A(f"| {cell['decay_val']:g} | {T_fin:.1e} | {kdens:.4f} | {p_hm.mean():.4f} "
          f"| {E_m1[i0]:+.3f} | {E_m2[i0]:+.3f} | {cond_gap:.3f} | {res['frac_converged']*100:.0f} |")
    A("")

    # --- графики M1 vs M2 и P(h=−1) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for res, st in zip(cells, stats):
        lab = f"decay={res['cell']['decay_val']:g}"
        ax1.plot(deg, st["E_m1"], "o-", ms=3, label=f"M1 {lab}")
        ax1.plot(deg, st["E_m2"], "s--", ms=3, label=f"M2 {lab}")
        ax2.plot(deg, st["p_hm"], "o-", ms=3, label=lab)
    ax1.axhline(0, color="gray", lw=0.5); ax1.set_xlabel("θ°"); ax1.set_ylabel("E")
    ax1.set_title("M1 (осевая) vs M2 (голономно-одетая)"); ax1.legend(fontsize=7); ax1.grid(alpha=0.2)
    ax2.set_xlabel("θ°"); ax2.set_ylabel("P(h=−1)"); ax2.set_title("Популяция h=−1")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(outdir / f"r5b_M1M2_h_{exp['mode']}.png", dpi=120); plt.close(fig)
    A(f"![M1/M2 и P(h=−1)](r5b_M1M2_h_{exp['mode']}.png)\n")

    # --- контроли ---
    A("## Контроли §4.2\n")
    A("| decay | маргиналы | ±-симм | воспроизв. |")
    A("|---|---|---|---|")
    for res, an in zip(cells, analyses):
        ctrl, repro = an["ctrl"], an["repro"]
        rr = (f"{'✅' if repro['passes'] else '❌'}") if repro else "—"
        A(f"| {res['cell']['decay_val']:g} | {'✅' if ctrl['all_marg_ok'] else '❌'} "
          f"| {'✅' if ctrl['all_sym_ok'] else '❌'} | {rr} |")
    A("")

    # --- вердикт по пре-регистрации ---
    A("## Вердикт по пре-регистрации\n")
    decays = [c["cell"]["decay_val"] for c in cells]
    kdns_s = ", ".join(f"{s['kdens']:.4f}" for s in stats)
    phm_s = ", ".join(f"{s['p_hm'].mean():.4f}" for s in stats)
    e_m1_s = ", ".join(f"{s['E_m1'][i0]:+.3f}" for s in stats)
    e_m2_s = ", ".join(f"{s['E_m2'][i0]:+.3f}" for s in stats)
    kink_pop = max(s["kdens"] for s in stats) > 1e-3
    A(f"**(a) отжиг населяет кинки:** decay {decays}: плотность кинков = [{kdns_s}]; "
      f"⟨P(h=−1)⟩ = [{phm_s}]. "
      f"{'Кинки/голономия НАСЕЛЕНЫ (>0).' if kink_pop else 'Кинки ПОЧТИ ОТСУТСТВУЮТ даже при отжиге.'} "
      "Направление тренда с decay — см. таблицу.")
    all_ferro_uncond = all(s["E_m1"][i0] > 0 for s in stats)
    A(f"\n**(b) безусловная E(θ) остаётся ферро:** "
      f"{'✅' if all_ferro_uncond else '❌'} (E_M1(0) = [{e_m1_s}]).")
    maxgap = float(np.nanmax([s["cond_gap"] for s in stats]))
    cond_signal = maxgap > 0.05
    A(f"\n**(c) условная E(θ|h) различается?** max по ячейкам |E(θ|h+)−E(θ|h−)| = {maxgap:.3f}. "
      + ("⚠️ ЕСТЬ различие ⇒ первый сигнал видимости спинорной структуры — паранойя и к архитектору."
         if cond_signal else
         "В пределах шума ⇒ спинорная структура невидима и в условной наблюдаемой."))
    A(f"\n**M2 (голономно-одетая):** E_M2(0) = [{e_m2_s}] против E_M1(0) = [{e_m1_s}]. "
      "Заметная антиферро-компонента в M2 появляется лишь при заметной популяции h=−1. "
      "M1 и M2 — РАЗНЫЕ определения наблюдаемой (не подгонка).")
    A("\n> Интерпретация — с архитектором (CLAUDE.md).")
    return "\n".join(L) + "\n"
