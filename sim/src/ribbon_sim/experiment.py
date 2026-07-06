"""Свип θ по ячейкам эксперимента, счётчики ветвей, сходимость, отчёт.

Поддержаны три формата конфига:
  - одиночная ячейка (R0): physics.k_e + dynamics.lr;
  - κ-сетка (R1): kappa_grid, k_e = κ·(N−1)·k_c, lr = 0.5/(k_e+k_c);
  - decay-сетка (R2): decay_grid, T0=1.0, k_e из лучшей ячейки R1.
"""

import time
from pathlib import Path

import jax
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
        base["steps"] = int(sm["steps"])
        base["seeds"] = list(sm["seeds"])
        thetas_deg = np.array(sm["theta_deg"], dtype=np.float64)

    cells = []
    if "kappa_grid" in cfg:  # R1
        for kappa in cfg["kappa_grid"]:
            k_e = float(kappa) * (N - 1) * k_c
            cells.append({**base, "k_e": k_e, "lr": 0.5 / (k_e + k_c),
                          "kappa": float(kappa), "label": f"κ={kappa:g}"})
    elif "decay_grid" in cfg:  # R2
        k_e = float(phys["k_e"])
        for d in cfg["decay_grid"]:
            cells.append({**base, "k_e": k_e, "lr": 0.5 / (k_e + k_c),
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


def run_experiment(cfg, mode="full", seed_override=None):
    cells, thetas, thetas_deg = build_cells(cfg, mode)
    conv_cfg = cfg["dynamics"].get("convergence")
    results = []
    for cell in cells:
        print(f"  [{cell['label']}] k_e={cell['k_e']:.3f} lr={cell['lr']:.2e} "
              f"steps={cell['steps']} T0={cell['T0']} decay={cell['decay']}")
        res = run_cell(cell, thetas, conv_cfg=conv_cfg, seed_override=seed_override)
        print(f"    → {res['elapsed_s']:.1f} c; сходимость max_frac={res['max_conv_frac']:.4f} "
              f"({'ok' if res['converged'] else 'NOT converged'}); "
              f"steps={res['steps_used']}"
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
    E = analysis.E_from_counts(counts_sum)
    ctrl = analysis.check_controls(counts_sum, thetas)
    repro = analysis.reproducibility(counts[0], counts[1]) if counts.shape[0] >= 2 else None
    cmp = analysis.compare_models(counts_sum, thetas)
    boot = analysis.bootstrap_p(counts_sum, thetas, n_boot=1000, seed=0)
    harm = analysis.harmonics(thetas, E)

    aics = {"пила": cmp["aic_saw"], "хорда-p": cmp["aic_chord_p"],
            "хорда-p=2": cmp["aic_chord_p2"]}
    best = min(aics, key=aics.get)
    amp = float(np.max(np.abs(E)))
    return {"E": E, "ctrl": ctrl, "repro": repro, "cmp": cmp, "boot": boot,
            "harm": harm, "aics": aics, "best": best, "amp": amp,
            "counts_sum": counts_sum}


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
        lbl = res["cell"]["label"].replace("=", "").replace("κ", "kappa").replace(".", "p")
        npz_payload[f"counts_{lbl}"] = res["counts"]
        # график E(θ) на ячейку
        seeds_E = [analysis.E_from_counts(res["counts"][si]) for si in range(res["counts"].shape[0])]
        plots.plot_E_curve(thetas, an["E"], outdir / f"E_{lbl}_{exp['mode']}.png",
                           seeds_E=seeds_E, title=f"{exp['name']} {res['cell']['label']} ({exp['mode']})")
    np.savez(outdir / f"counts_{exp['mode']}.npz", **npz_payload)

    # Сводный график всех ячеек.
    _plot_all_cells(exp, analyses, outdir / f"E_all_{exp['mode']}.png")

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
        sgn = "ферро (+)" if an["cmp"]["corr_sign"] < 0 else "антиферро (−)"
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
    sgn = "ферро E(0)=+1" if cmp["corr_sign"] < 0 else "антиферро E(0)=−1"
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
            sgn = "ферро" if an["cmp"]["corr_sign"] < 0 else "антиферро"
            p2_flag = "← p̂≈2 (!)" if abs(an["cmp"]["p_hat"] - 2.0) < 0.15 else ""
            A(f"- **{res['cell']['label']}**: амплитуда {an['amp']:.3f}, {sgn}, лучшая модель "
              f"{an['best']}, p̂={an['cmp']['p_hat']:.3f} {p2_flag}")
        anyp2 = any(abs(an["cmp"]["p_hat"] - 2.0) < 0.15 for an in analyses)
        anyferro = any(an["cmp"]["corr_sign"] < 0 for an in analyses)
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
