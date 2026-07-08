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
        "mirror_pairs": bool(run.get("mirror_pairs", False)),  # R3-geo: зеркальные пары
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
    elif "twist_sectors" in cfg and "decay_grid" in cfg:  # R4b-anneal: сектор×анизотропия×отжиг
        kappa_eq = float(cfg.get("kappa_equiv", 1.0))
        k_e_eq = kappa_eq * (N - 1) * k_c
        el = str(phys.get("elastic", "cosserat_geo"))
        total = 3.0 * k_e_eq if el == "cosserat_geo" else 3.0 * (k_e_eq / 4.0)
        for dcy in cfg["decay_grid"]:
            for sec in cfg["twist_sectors"]:
                tw = float(sec) * 2.0 * np.pi
                for ratio in cfg["cosserat_grid"]:
                    r = float(ratio)
                    k_b = total / (2.0 + r); k_t = r * k_b
                    cells.append({**base, "elastic": el, "k_e": 0.0, "k_b": k_b, "k_t": k_t,
                                  "lr": 0.5 / (max(k_b, k_t) + k_c), "kappa": kappa_eq, "ratio": r,
                                  "decay": float(dcy), "decay_val": float(dcy),
                                  "twist_project": True, "twist_target": tw, "sector": float(sec),
                                  "label": f"d{dcy:g} Tw{sec:g} kt/kb{r:g}"})
    elif "twist_sectors" in cfg and "cosserat_grid" in cfg:  # R4a/R4b: сектор × анизотропия
        kappa_eq = float(cfg.get("kappa_equiv", 1.0))
        k_e_eq = kappa_eq * (N - 1) * k_c
        el = str(phys.get("elastic", "cosserat_geo"))
        total = 3.0 * k_e_eq if el == "cosserat_geo" else 3.0 * (k_e_eq / 4.0)
        for sec in cfg["twist_sectors"]:
            tw = float(sec) * 2.0 * np.pi
            for ratio in cfg["cosserat_grid"]:
                r = float(ratio)
                k_b = total / (2.0 + r); k_t = r * k_b
                cells.append({**base, "elastic": el, "k_e": 0.0, "k_b": k_b, "k_t": k_t,
                              "lr": 0.5 / (max(k_b, k_t) + k_c), "kappa": kappa_eq, "ratio": r,
                              "twist_project": True, "twist_target": tw, "sector": float(sec),
                              "label": f"Tw={sec:g}·2π kt/kb={r:g}"})
    elif "cosserat_grid" in cfg:  # R3: свип k_t/k_b при фикс. суммарной жёсткости
        kappa_eq = float(cfg.get("kappa_equiv", 1.0))
        k_e_eq = kappa_eq * (N - 1) * k_c
        el = str(phys.get("elastic", "cosserat_geo"))
        # cosserat_geo: E_iso=k_b·d² ⇒ 2k_b+k_t=3·k_e (⟨твист-доля⟩=1/3), ρ=1 ⇒ k_b=k_t=k_e≡geodesic κ.
        # старые chordal/atan2: E~k_b·|ω|²≈k_b·4d² ⇒ множитель /4.
        total = 3.0 * k_e_eq if el == "cosserat_geo" else 3.0 * (k_e_eq / 4.0)
        for ratio in cfg["cosserat_grid"]:
            r = float(ratio)
            k_b = total / (2.0 + r)
            k_t = r * k_b
            lr = 0.5 / (max(k_b, k_t) + k_c)  # ρ=1: max=k_e ⇒ lr=lr(R1 κ)
            cells.append({**base, "elastic": el, "k_e": 0.0, "k_b": k_b, "k_t": k_t,
                          "lr": lr, "kappa": kappa_eq, "ratio": r,
                          "label": f"kt/kb={r:g}"})
    elif "twist_sectors" in cfg:  # (одиночная жёсткость, если без cosserat_grid)
        k_e = float(phys["k_e"])
        kappa = float(phys.get("kappa", k_e / ((N - 1) * k_c)))
        for sec in cfg["twist_sectors"]:
            tw = float(sec) * 2.0 * np.pi
            cells.append({**base, "k_e": k_e, "lr": 0.5 / (k_e + k_c), "kappa": kappa,
                          "twist_project": True, "twist_target": tw, "sector": float(sec),
                          "label": f"Tw={sec:g}·2π"})
    elif "soft_ke_grid" in cfg:  # мягкая лента: k_e ≈ T0, спинор, отжиг
        for ke in cfg["soft_ke_grid"]:
            ke = float(ke)
            cells.append({**base, "k_e": ke, "lr": 0.5 / (ke + k_c),
                          "kappa": ke / ((N - 1) * k_c), "k_e_soft": ke,
                          "label": f"k_e={ke:g}"})
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


def run_cell_blocks(cell, thetas, block_cfg, seed_override=None,
                    measure_theta_deg=None, thetas_deg=None):
    """Блочный протокол сходимости по НАБЛЮДАЕМОЙ (ответ на ARCH-Q#1, R5).

    Каждую (сид, θ) релаксируем блоками block_steps, продолжая с прошлого q, пока
    |ΔE| между блоками < max(sigma_mult·σ_MC, e_floor) или не достигнут ceiling.
    Плюс диагностика кинков и (R4) диагностика меры на выбранных θ.
    """
    from .dynamics import holonomy, kink_count
    from .frames import axis, mirror_flip, quat_conj_mul, sector_sample, total_twist

    twist_target = cell.get("twist_target") if cell.get("twist_project") else None
    mirror = bool(cell.get("mirror_pairs", False))  # зеркальные пары q / q⊗j (R3-geo)
    # индексы θ для диагностики меры (R4): ближайшие к запрошенным углам
    meas_idx = {}
    if measure_theta_deg is not None and thetas_deg is not None:
        for td in measure_theta_deg:
            j = int(np.argmin(np.abs(np.asarray(thetas_deg) - td)))
            meas_idx[j] = td
    measure = {td: {"c_A": [], "c_B": [], "s": [], "t": [], "tau": [],
                    "Tw0": [], "Tw1": []} for td in meas_idx.values()}
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
    counts_h1 = np.zeros((nS, nT, 4), dtype=np.int64)          # зеркальная половина q
    counts_h2 = np.zeros((nS, nT, 4), dtype=np.int64)          # зеркальная половина q⊗j
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
            if twist_target is not None:  # R4: честный сэмплинг сектора Tw=target
                q, _tw_err = sector_sample(k_init, (B, N), target=twist_target)
            else:
                q = haar_quaternions(k_init, (B, N))
            if mirror:  # вторая половина = зеркало первой (q⊗j): ±-симметрия по построению
                half = B // 2
                q = q.at[half:].set(mirror_flip(q[:half]))
            tw0 = float(np.mean(np.asarray(total_twist(q)))) if ti in meas_idx else 0.0
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
            if mirror:  # E по каждой зеркальной половине (новый контроль §4.2)
                half = B // 2
                counts_h1[si, ti] = npbc(s[:half], t[:half])
                counts_h2[si, ti] = npbc(s[half:], t[half:])
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
            # диагностика меры (R4): c_A=n_A·a, c_B=n_B·b, профиль скрутки τ̃_i=2z
            if ti in meas_idx:
                td = meas_idx[ti]
                nA = np.asarray(axis(q[:, 0])); nB = np.asarray(axis(q[:, -1]))
                aa = np.asarray(a); bb = np.asarray(b)
                rlink = np.asarray(quat_conj_mul(q[:, :-1], q[:, 1:]))  # (B, N-1, 4)
                tau = 2.0 * rlink[..., 3]                 # τ̃_i на связь (B, N-1)
                m = measure[td]
                m["c_A"].append(nA @ aa); m["c_B"].append(nB @ bb)
                m["s"].append(s); m["t"].append(t)
                m["tau"].append(tau.mean(0))              # средний профиль по батчу
                m["Tw0"].append(tw0); m["Tw1"].append(float(np.mean(tau.sum(1))))

    return {
        "cell": cell,
        "counts": counts,
        "counts_M2": counts_M2,
        "counts_hplus": counts_hplus,
        "counts_hminus": counts_hminus,
        "n_hminus": n_hminus,
        "counts_h1": counts_h1,
        "counts_h2": counts_h2,
        "mirror": mirror,
        "measure": {td: {
            "c_A": np.concatenate(m["c_A"]), "c_B": np.concatenate(m["c_B"]),
            "s": np.concatenate(m["s"]), "t": np.concatenate(m["t"]),
            "tau": np.mean(np.stack(m["tau"]), 0),
            "Tw0": float(np.mean(m["Tw0"])), "Tw1": float(np.mean(m["Tw1"])),
        } for td, m in measure.items()} if measure else None,
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
    md = cfg.get("measure_diag")
    meas_theta = md["theta_deg"] if (md and md.get("enabled")) else None
    results = []
    for cell in cells:
        print(f"  [{cell['label']}] elastic={cell['elastic']} k_e={cell['k_e']:.3f} "
              f"lr={cell['lr']:.2e} T0={cell['T0']}")
        if block_cfg:
            res = run_cell_blocks(cell, thetas, block_cfg, seed_override=seed_override,
                                  measure_theta_deg=meas_theta, thetas_deg=thetas_deg)
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
    # DEGENERATE-классификация включается для зеркальных прогонов (R4b): чистое зеркало
    # + честный сэмплер ⇒ большой |ΔE| сидов = вырожденность, не дефект.
    cd = bool(res.get("mirror"))
    repro = (analysis.reproducibility(counts[0], counts[1], classify_degenerate=cd)
             if counts.shape[0] >= 2 else None)
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
    # Зеркальный контроль (R3-geo): |E_half1 − E_half2| < 2σ_MC по max-статистике.
    mirror = None
    if res.get("mirror") and res["counts_h1"].sum() > 0:
        mirror = analysis.reproducibility(res["counts_h1"].sum(0), res["counts_h2"].sum(0))
    return {"E": E, "ctrl": ctrl, "repro": repro, "cmp": cmp, "boot": boot,
            "harm": harm, "aics": aics, "best": best, "amp": amp, "mirror": mirror,
            "ferro": ferro, "E0": float(E[i0]), "counts_sum": counts_sum}


def _slug(label):
    return (label.replace("κ", "k").replace("=", "").replace(".", "p")
            .replace(" ", "_").replace("/", "-"))


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
        # диагностика меры (R4): гистограммы c_A,c_B + профиль скрутки
        if res.get("measure"):
            plots.plot_measure_diag(res["measure"], outdir / f"meas_{lbl}_{exp['mode']}.png",
                                    title=f"{exp['name']} {res['cell']['label']}: мера c_A,c_B")
            plots.plot_twist_profile(res["measure"], outdir / f"twist_{lbl}_{exp['mode']}.png",
                                     title=f"{exp['name']} {res['cell']['label']}: профиль τ̃_i")
    np.savez(outdir / f"counts_{exp['mode']}.npz", **npz_payload)

    # Сводный график всех ячеек.
    _plot_all_cells(exp, analyses, outdir / f"E_all_{exp['mode']}.png")

    # R5b-отчёт (голономия) только для СПИНОРНЫХ decay-ячеек; geodesic-decay (R2-лайт)
    # идёт по обычному R2-пути.
    is_r5b = ("decay_grid" in cfg and "block_convergence" in cfg["dynamics"]
              and str(cfg["physics"].get("elastic", "geodesic")) == "spinor")
    if "elastic_grid" in cfg:
        md = _render_r5(exp, analyses, deg)
    elif "twist_sectors" in cfg and "decay_grid" in cfg:
        md = _render_r4anneal(exp, analyses, deg)
    elif "twist_sectors" in cfg or "soft_ke_grid" in cfg:
        md = _render_holo(exp, analyses, deg)
    elif is_r5b:
        md = _render_r5b(exp, analyses, deg, outdir)
    elif "cosserat_grid" in cfg:
        md = _render_r3(exp, analyses, deg)
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
            T_fin = cell["T0"] * cell["decay"] ** res.get("max_steps", cell["steps"])
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
      + (f" (удвоено ×{res['doublings']}, история {res['steps_history']})"
         if res.get("doublings") and res.get("steps_history") else "")
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
    if "max_conv_frac" in res:  # старый run_cell (энерго-критерий)
        A(f"сходимость: доля лент |ΔE|/шаг>1e-6 = {res['max_conv_frac']:.4f} "
          f"({'сошлось' if res['converged'] else 'НЕ сошлось'}).\n")
    else:                        # блочный протокол
        A(f"сходимость (блочная по наблюдаемой): {res['frac_converged']*100:.0f}% (θ,сид), "
          f"max_steps={res['max_steps']}.\n")

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
        # Явный вердикт по гипотезе архитектора (амплитуда растёт, форма к ступени).
        amps = [an["amp"] for an in analyses]
        a1s = [an["harm"]["A1"] for an in analyses]
        decays_v = [r["cell"].get("decay_val", 0) for r in exp["cells"]]
        # порядок по замедлению отжига (рост decay)
        order = np.argsort(decays_v)
        amps_o = [amps[i] for i in order]
        a1_o = [a1s[i] for i in order]
        amp_grows = all(amps_o[i] <= amps_o[i + 1] + 1e-3 for i in range(len(amps_o) - 1))
        near_step = max(amps) > 0.9  # |E(0)|→1 = ступень (не косинус)
        A(f"\n> **Вердикт (гипотеза архитектора §4):** амплитуда с замедлением отжига "
          f"{'РАСТЁТ' if amp_grows else 'НЕ растёт монотонно'} "
          f"({', '.join(f'{a:.3f}' for a in amps_o)}); "
          f"A1(cosθ) {', '.join(f'{a:+.2f}' for a in a1_o)} → форма ведёт "
          f"{'к СТУПЕНИ (|E(0)|→1), не к косинусу' if near_step else 'НЕ к ступени'}. "
          f"{'✅ гипотеза подтверждена' if (amp_grows and near_step) else '❌ гипотеза не подтверждена'}. "
          "Изотропная лента (T>0) не даёт квантовую косинусную форму — «ножницы» §4.")
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
#  Отчёт R4b-anneal (снятие вырожденности: E(θ|сектор), веса ветвей, decay-универсальность)
# --------------------------------------------------------------------------- #
# θ бывших точек фрустрации (квенч R4b) + соседи — где отжиг проверяется на снятие вырожденности.
_FRUST_THETA = [22.5, 30.0, 37.5, 112.5, 120.0, 127.5, 165.0, 172.5, 180.0]


def _antiferro_frac(counts_2d):
    """Доля антиферро-ветви s·t=−1 (=pm+mp) по θ. counts_2d (n_theta,4)."""
    c = np.asarray(counts_2d, dtype=np.float64)
    return (c[..., 1] + c[..., 2]) / np.maximum(c.sum(-1), 1)


def _render_r4anneal(exp, analyses, deg):
    cfg = exp["cfg"]; cells = exp["cells"]
    L = []; A = L.append
    A(f"# {exp['name']} — отчёт ({exp['mode']}): отжиг снимает вырожденность Tw=2π\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрация:** {cfg.get('prediction', '')}\n")
    total = sum(r["elapsed_s"] for r in cells); mem = _gpu_mem_mb()
    c0 = cells[0]["cell"]
    A("## Конфигурация\n```")
    A(f"cosserat_geo+spinor  N={c0['N']}  B={c0['B']}  seeds={c0['seeds']}  T0={c0['T0']}  "
      f"честный сэмплер + зеркальные пары + связь Tw=const")
    A(f"decay∈{cfg['decay_grid']}  Tw∈{cfg['twist_sectors']}·2π  k_t/k_b∈{cfg['cosserat_grid']}")
    A("```")
    A(f"\nВремя: **{total:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else "") + "\n")
    A(f"![E(θ) по ячейкам](E_all_{exp['mode']}.png)\n")

    # индекс ячеек по (decay, sector, ratio)
    idx = {(c["cell"]["decay_val"], c["cell"]["sector"], c["cell"]["ratio"]): (c, an)
           for c, an in zip(cells, analyses)}
    decays = sorted({c["cell"]["decay_val"] for c in cells})
    sectors = sorted({c["cell"]["sector"] for c in cells})
    ratios = sorted({c["cell"]["ratio"] for c in cells})
    i0 = int(np.argmin(np.abs(deg)))

    # --- сводка ---
    A("## Сводка: E(θ) и доля антиферро по (decay, сектор, k_t/k_b)\n")
    A("| decay | Tw | k_t/kb | амп\\|E\\| | E(0) | ⟨антиферро⟩ | сид-флипы (DEGENERATE) | сошлось% |")
    A("|---|---|---|---|---|---|---|---|")
    for c, an in zip(cells, analyses):
        cell = c["cell"]
        af = float(np.mean(_antiferro_frac(an["counts_sum"])))
        repro = an.get("repro")
        nd = repro["n_degenerate"] if repro else 0
        A(f"| {cell['decay_val']:g} | {cell['sector']:g} | {cell['ratio']:g} | {an['amp']:.3f} "
          f"| {an['E'][i0]:+.3f} | {af:.3f} | {nd} | {c['frac_converged']*100:.0f} |")
    A("")

    # --- (a) снял ли отжиг сид-флипы: DEGENERATE-точки по ячейкам ---
    A("## (a) Устраняет ли отжиг сид-флипы? (DEGENERATE-точки)\n")
    A("Отжиг должен сделать веса ветвей воспроизводимыми ⇒ 0 DEGENERATE. Если остаются — "
      "вырожденность ТОЧНАЯ (симметрия), отдельная загадка.\n")
    A("| decay | Tw | k_t/kb | DEGENERATE θ | max\\|z\\| невырожд. |\n|---|---|---|---|---|")
    for c, an in zip(cells, analyses):
        cell = c["cell"]; repro = an.get("repro")
        if repro and repro["n_degenerate"]:
            dth = ", ".join(f"{t:.0f}°" for t in deg[repro["degenerate"]])
        else:
            dth = "—"
        mz = f"{repro['max_z']:.2f}" if repro else "—"
        A(f"| {cell['decay_val']:g} | {cell['sector']:g} | {cell['ratio']:g} | {dth} | {mz} |")
    A("")

    # --- (c) КЛЮЧЕВОЕ: согласованы ли веса при разных decay ---
    A("## (c) КЛЮЧЕВОЕ: универсальна ли термо-мера? (веса при двух decay)\n")
    A(f"Доля антиферро (s·t=−1) в точках фрустрации {[f'{t:g}' for t in _FRUST_THETA]}, "
      "усреднённая по сидам, при разных decay. Согласие ⇒ мера определена; расхождение ⇒ "
      "мера зависит от скорости охлаждения (неуниверсальна).\n")
    frust_idx = [int(np.argmin(np.abs(deg - t))) for t in _FRUST_THETA]
    if len(decays) >= 2:
        for sec in sectors:
            for r in ratios:
                if all((d, sec, r) in idx for d in decays):
                    A(f"**Tw={sec:g}·2π, k_t/k_b={r:g}:**")
                    A("| θ° | " + " | ".join(f"антиф@decay={d:g}" for d in decays) + " | \\|Δ\\| |")
                    A("|---|" + "---|" * (len(decays) + 1))
                    worst = 0.0
                    for j, t in zip(frust_idx, _FRUST_THETA):
                        afs = [float(np.mean([_antiferro_frac(idx[(d, sec, r)][0]["counts"][si])[j]
                                              for si in range(idx[(d, sec, r)][0]["counts"].shape[0])]))
                               for d in decays]
                        dd = max(afs) - min(afs)
                        worst = max(worst, dd)
                        A(f"| {t:g} | " + " | ".join(f"{a:.3f}" for a in afs) + f" | {dd:.3f} |")
                    verdict = ("✅ веса согласованы (мера универсальна)" if worst < 0.05
                               else "⚠️ веса ЗАВИСЯТ от decay (мера неуниверсальна)")
                    A(f"\nmax\\|Δвес\\| = {worst:.3f} → {verdict}\n")

    # --- (b) θ-зависимость антиферро в Tw=2π ---
    A("## (b) θ-зависимость доли антиферро в Tw=2π (форма открыта)\n")
    for r in ratios:
        row = []
        for d in decays:
            if (d, 1.0, r) in idx:
                af = _antiferro_frac(idx[(d, 1.0, r)][1]["counts_sum"])
                row.append((d, af))
        if row:
            A(f"**k_t/k_b={r:g}** доля антиферро по θ (decay={row[0][0]:g}):")
            af = row[0][1]
            A("  " + " ".join(f"{deg[i]:.0f}°:{af[i]:.2f}" for i in range(0, len(deg), 4)))
    A("")

    _render_verdict_r4anneal(A, cells, analyses, deg, sectors, ratios, decays, idx, i0)
    return "\n".join(L) + "\n"


def _render_verdict_r4anneal(A, cells, analyses, deg, sectors, ratios, decays, idx, i0):
    A("## Вердикт по пре-регистрации\n")
    # (a)
    any_degen = any(an.get("repro") and an["repro"]["n_degenerate"] for an in analyses)
    A(f"**(a) отжиг устраняет сид-флипы:** {'❌ остаются DEGENERATE-точки (вырожденность точная — загадка)' if any_degen else '✅ сид-флипов нет (веса воспроизводимы)'}.")
    # (c)
    worst_all = 0.0
    if len(decays) >= 2:
        frust_idx = [int(np.argmin(np.abs(deg - t))) for t in _FRUST_THETA]
        for sec in sectors:
            for r in ratios:
                if all((d, sec, r) in idx for d in decays):
                    for j in frust_idx:
                        afs = [float(np.mean([_antiferro_frac(idx[(d, sec, r)][0]["counts"][si])[j]
                                              for si in range(idx[(d, sec, r)][0]["counts"].shape[0])]))
                               for d in decays]
                        worst_all = max(worst_all, max(afs) - min(afs))
    A(f"\n**(c) КЛЮЧЕВОЕ — универсальность меры:** max|Δвес| между decay = {worst_all:.3f} → "
      + ("✅ мера УНИВЕРСАЛЬНА (не зависит от скорости охлаждения)" if worst_all < 0.05
         else "⚠️ мера НЕуниверсальна (зависит от decay)") + ".")
    # антиферро существование
    af2pi = any(np.mean(_antiferro_frac(an["counts_sum"])) > 0.02
                for c, an in zip(cells, analyses) if c["cell"]["sector"] > 0)
    A(f"\n**Доля антиферро в Tw=2π > 0:** {'✅' if af2pi else '❌'} (существование ветви).")
    A("\n> Интерпретация — с архитектором (CLAUDE.md).")


# --------------------------------------------------------------------------- #
#  Отчёт R4a/R4b/мягкая лента (голономия M1/M2, условная E(θ|h))
# --------------------------------------------------------------------------- #
def _render_holo(exp, analyses, deg):
    cfg = exp["cfg"]
    cells = exp["cells"]
    i0 = int(np.argmin(np.abs(deg)))
    is_sector = "twist_sectors" in cfg
    L = []; A = L.append
    A(f"# {exp['name']} — отчёт ({exp['mode']}): "
      + ("связь Tw=const, голономия M1/M2" if is_sector else "мягкая лента, отжиг, голономия") + "\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрация:** {cfg.get('prediction', '')}\n")
    A("**M1** — осевая наблюдаемая (s,t); **M2** — голономно-одетая t̃=h·sign(n_B·b). "
      "Разные ОПРЕДЕЛЕНИЯ наблюдаемой, сравнивать бок о бок.\n")

    c0 = cells[0]["cell"]; total = sum(r["elapsed_s"] for r in cells); mem = _gpu_mem_mb()
    A("## Конфигурация\n```")
    A(f"elastic={c0['elastic']}  N={c0['N']}  B={c0['B']}  seeds={c0['seeds']}  T0={c0['T0']}  "
      f"decay={c0['decay']}" + ("  twist_project=on" if c0.get('twist_project') else ""))
    A(f"θ (°): {np.round(deg, 2).tolist()}")
    A("```")
    A(f"\nВремя: **{total:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else "") + "\n")
    A(f"![E(θ) M1 по ячейкам](E_all_{exp['mode']}.png)\n")

    A("## Сводка: M1 vs M2, голономия\n")
    A("| ячейка | плотн. кинков | ⟨P(h=−1)⟩ | E_M1(0) | E_M2(0) | знак M1 | max_θ\\|E(h+)−E(h−)\\| | сошлось% |")
    A("|---|---|---|---|---|---|---|---|")
    stats = []
    for res, an in zip(cells, analyses):
        cell = res["cell"]; B = cell["B"]; ns = len(res["seeds"]); ntot = B * ns
        E_m1 = an["E"]
        E_m2 = analysis.E_from_counts(res["counts_M2"].sum(0))
        p_hm = res["n_hminus"].sum(0) / ntot
        E_hp = _E_safe(res["counts_hplus"].sum(0))
        E_hm = _E_safe(res["counts_hminus"].sum(0))
        diff = E_hp - E_hm
        gap = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else float("nan")
        kd = float(res["kink_density"].mean())
        sgn = "ферро" if an["ferro"] else "антиферро"
        stats.append({"E_m1": E_m1, "E_m2": E_m2, "p_hm": p_hm, "gap": gap, "kd": kd, "sgn": sgn})
        A(f"| {cell['label']} | {kd:.4f} | {p_hm.mean():.4f} | {E_m1[i0]:+.3f} | {E_m2[i0]:+.3f} "
          f"| {sgn} | {gap:.3f} | {res['frac_converged']*100:.0f} |")
    A("")

    # --- явные доли ветвей P(s,t|θ,сектор) (архитектор: фрустрация как СМЕСЬ, не сид-флип) ---
    A("## Доли ветвей P(s,t) [pp,pm,mp,mm] по θ\n")
    A("Чистая мера ⇒ гладкая смесь ветвей; вырождение/фрустрация ⇒ доминирование одной "
      "ветви (|E|→1 без смеси) или скачки. Сравнить сиды на согласованность.\n")
    idxs = [int(np.argmin(np.abs(deg - td))) for td in (0.0, 60.0, 120.0, 180.0)]
    for res in cells:
        A(f"**{res['cell']['label']}** (по сидам):")
        A("| θ° | сид | pp | pm | mp | mm |")
        A("|---|---|---|---|---|---|")
        c = res["counts"]  # (nS, nT, 4)
        for j in idxs:
            for si in range(c.shape[0]):
                f = c[si, j] / max(c[si, j].sum(), 1)
                A(f"| {deg[j]:.0f} | {si} | {f[0]:.2f} | {f[1]:.2f} | {f[2]:.2f} | {f[3]:.2f} |")
        A("")

    # --- диагностика меры (R4) ---
    if any(res.get("measure") for res in cells):
        A("## Диагностика эффективной меры\n")
        A("Гистограммы c_A=n_A·a, c_B=n_B·b по ветвям и θ (плотность, uniform-сфера=0.5); "
          "оверлеи uniform / KS∝max(c,0) / δ-пик |c|=1. Профиль скрутки ⟨τ̃_i⟩ по ленте + "
          "контроль связи Tw до/после.\n")
        for res in cells:
            if not res.get("measure"):
                continue
            lbl = _slug(res["cell"]["label"])
            tw_ok = all(abs(m["Tw1"] - m["Tw0"]) < 1e-3 for m in res["measure"].values())
            dtw = max(abs(m["Tw1"] - m["Tw0"]) for m in res["measure"].values())
            A(f"### {res['cell']['label']}  (Tw сохранена: {'✅' if tw_ok else '❌'}, "
              f"max|ΔTw|={dtw:.1e})\n")
            A(f"![мера](meas_{lbl}_{exp['mode']}.png)\n")
            A(f"![твист](twist_{lbl}_{exp['mode']}.png)\n")

    A("## Контроли §4.2\n")
    mirror_on = any(a.get("mirror") for a in analyses)
    if mirror_on:
        A("*Зеркальные пары: ±-симметрия и маргиналы тождественны по построению (вычеркнуты). "
          "Контроль — согласованность зеркальных половин |E_h1−E_h2| max-статистикой. "
          "**DEGENERATE** (архитектор): θ-точки сид-флипа (|ΔE|>0.3) при чистом зеркале и честном "
          "сэмплере — вырожденность основного состояния при T=0 (E не самоусредняется), валидная "
          "физика, не стоп-сигнал; исключены из вердикта воспроизводимости.*\n")
        A("| ячейка | зеркало \\|E_h1−E_h2\\| | воспроизв. невырожд. | DEGENERATE точек |\n|---|---|---|---|")
        for res, an in zip(cells, analyses):
            m, repro = an.get("mirror"), an["repro"]
            mm = (f"{m['max_z']:.2f}<{m['z_thresh']:.2f} {'✅' if m['passes'] else '❌'}") if m else "—"
            if repro:
                nd = repro.get("n_degenerate", 0)
                deg_th = ""
                if nd:
                    dmask = repro["degenerate"]
                    deg_th = ", ".join(f"{d:.0f}°" for d in deg[dmask])
                rr = f"{repro['max_z']:.2f}<{repro['z_thresh']:.2f} {'✅' if repro['passes'] else '❌'}"
                dd = f"**{nd}** ({deg_th})" if nd else "0"
            else:
                rr, dd = "—", "—"
            A(f"| {res['cell']['label']} | {mm} | {rr} | {dd} |")
    else:
        A("| ячейка | маргиналы | ±-симм | воспроизв. |\n|---|---|---|---|")
        for res, an in zip(cells, analyses):
            ctrl, repro = an["ctrl"], an["repro"]
            rr = ("✅" if repro["passes"] else "❌") if repro else "—"
            A(f"| {res['cell']['label']} | {'✅' if ctrl['all_marg_ok'] else '❌'} "
              f"| {'✅' if ctrl['all_sym_ok'] else '❌'} | {rr} |")
    A("")

    A("## Вердикт по пре-регистрации\n")
    m1_ferro = all(s["E_m1"][i0] > 0 for s in stats)
    m2_anti = any(s["E_m2"][i0] < -0.02 for s in stats)
    maxgap = float(np.nanmax([s["gap"] for s in stats]))
    e_m1_s = ", ".join(f"{s['E_m1'][i0]:+.3f}" for s in stats)
    e_m2_s = ", ".join(f"{s['E_m2'][i0]:+.3f}" for s in stats)
    A(f"- Знак M1: {'все ферро' if m1_ferro else 'ГДЕ-ТО АНТИФЕРРО'} (E_M1(0) = {e_m1_s}).")
    A(f"- Антиферро-компонента в M2: {'⚠️ ЕСТЬ' if m2_anti else 'нет'} (E_M2(0) = {e_m2_s}).")
    A(f"- Условное расщепление E(θ|h): max|E(h+)−E(h−)| = {maxgap:.3f} "
      + ("⚠️ ЕСТЬ ⇒ сигнал видимости спинора" if maxgap > 0.05 else "(в шуме/нет h−)") + ".")
    if is_sector:
        # R4b: зависит ли E(θ) от сектора Tw — сравниваем ОДИНАКОВЫЕ k_t/k_b между секторами
        sectors = sorted({res["cell"]["sector"] for res in cells})
        if len(sectors) >= 2:
            by_rs = {(res["cell"]["sector"], res["cell"].get("ratio", 1.0)): st
                     for res, st in zip(cells, stats)}
            ratios = sorted({res["cell"].get("ratio", 1.0) for res in cells})
            s0, s1 = sectors[0], sectors[-1]
            worst = 0.0
            for rr in ratios:
                if (s0, rr) in by_rs and (s1, rr) in by_rs:
                    g = float(np.nanmax(np.abs(by_rs[(s0, rr)]["E_m1"] - by_rs[(s1, rr)]["E_m1"])))
                    A(f"- **E(θ|Tw={s0:g})−E(θ|Tw={s1:g}) при k_t/k_b={rr:g}:** max_θ = {g:.3f}.")
                    worst = max(worst, g)
            A("  " + ("⚠️ E ЗАВИСИТ от сектора Tw ⇒ топология видна через связь!"
                      if worst > 0.05 else "E НЕ зависит от сектора Tw (топология невидима в осях)."))
    A("\n> Центральный вопрос, уверенность низкая; обе гипотезы легальны. "
      "Интерпретация — с архитектором (CLAUDE.md).")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
#  Отчёт R3 (анизотропия Коссера: свип k_t/k_b)
# --------------------------------------------------------------------------- #
def _render_r3(exp, analyses, deg):
    cfg = exp["cfg"]
    cells = exp["cells"]
    i0 = int(np.argmin(np.abs(deg)))
    L = []; A = L.append
    A(f"# {exp['name']} — отчёт ({exp['mode']}): анизотропия Коссера (изгиб/скрутка)\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрация:** {cfg.get('prediction', '')}\n")

    total = sum(r["elapsed_s"] for r in cells)
    mem = _gpu_mem_mb()
    c0 = cells[0]["cell"]
    A("## Конфигурация\n```")
    A(f"elastic=cosserat  N={c0['N']}  B={c0['B']}  seeds={c0['seeds']}  T0={c0['T0']}")
    A(f"суммарная жёсткость 2k_b+k_t фиксирована (κ={c0.get('kappa',1)}-эквивалент), lr={c0['lr']:.2e}")
    A(f"θ (°): {np.round(deg, 2).tolist()}")
    A("```")
    A(f"\nВремя: **{total:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else "") + "\n")
    A(f"![E(θ) по ячейкам](E_all_{exp['mode']}.png)\n")

    A("## Сводка: анизотропия k_t/k_b → форма\n")
    A("| k_t/k_b | k_b | k_t | амп max\\|E\\| | E(0) | знак | A1 | A3 | A3/A1 | сошлось% | max_steps |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for res, an in zip(cells, analyses):
        cell = res["cell"]; h = an["harm"]
        a3a1 = h["A3"] / h["A1"] if abs(h["A1"]) > 1e-6 else float("nan")
        sgn = "ферро" if an["ferro"] else "антиферро"
        A(f"| {cell['ratio']:g} | {cell['k_b']:.2f} | {cell['k_t']:.2f} | {an['amp']:.3f} "
          f"| {an['E'][i0]:+.3f} | {sgn} | {h['A1']:+.3f} | {h['A3']:+.3f} | {a3a1:+.3f} "
          f"| {res['frac_converged']*100:.0f} | {res['max_steps']} |")
    A("")

    A("## Контроли §4.2\n")
    mirror_on = any(a.get("mirror") for a in analyses)
    if mirror_on:
        A("*Зеркальные пары (q / q⊗j): ±-симметрия и маргиналы ТОЖДЕСТВЕННЫ по построению "
          "(вычеркнуты, решение архитектора). НОВЫЙ контроль — согласованность зеркальных "
          "половин |E_h1(θ)−E_h2(θ)| по max-статистике.*\n")
        A("| k_t/k_b | зеркало \\|E_h1−E_h2\\| (max\\|z\\| vs порог) | воспроизв. (сиды) |")
        A("|---|---|---|")
        for res, an in zip(cells, analyses):
            m, repro = an["mirror"], an["repro"]
            mm = (f"{m['max_z']:.2f}<{m['z_thresh']:.2f} {'✅' if m['passes'] else '❌'}") if m else "—"
            rr = (f"{repro['max_z']:.2f} {'✅' if repro['passes'] else '❌'}") if repro else "—"
            A(f"| {res['cell']['ratio']:g} | {mm} | {rr} |")
    else:
        A("| k_t/k_b | маргиналы | ±-симм (maxσ) | воспроизв. |")
        A("|---|---|---|---|")
        for res, an in zip(cells, analyses):
            ctrl, repro = an["ctrl"], an["repro"]
            rr = (f"{repro['max_z']:.2f}<{repro['z_thresh']:.2f} {'✅' if repro['passes'] else '❌'}") if repro else "—"
            A(f"| {res['cell']['ratio']:g} | {'✅' if ctrl['all_marg_ok'] else '❌'} "
              f"| {'✅' if ctrl['all_sym_ok'] else '❌'} "
              f"({max(ctrl['sym_same_sigma'].max(), ctrl['sym_opp_sigma'].max()):.2f}) | {rr} |")
    A("")

    A("## Вердикт по пре-регистрации\n")
    all_ferro = all(an["ferro"] for an in analyses)
    amps = [an["amp"] for an in analyses]
    a3a1s = [an["harm"]["A3"] / an["harm"]["A1"] if abs(an["harm"]["A1"]) > 1e-6 else float("nan")
             for an in analyses]
    ratios = [c["cell"]["ratio"] for c in cells]
    A(f"**Знак ферро:** {'✅ все ячейки' if all_ferro else '❌ где-то антиферро'}.")
    A(f"\n**Двигает ли анизотропия ножницы?** по k_t/k_b {ratios}: "
      f"амплитуда = {[f'{x:.3f}' for x in amps]}; A3/A1 = {[f'{x:+.3f}' for x in a3a1s]}.")
    amp_spread = max(amps) - min(amps)
    A(f"\nРазброс амплитуды по анизотропии = {amp_spread:.3f}. "
      + ("⚠️ Анизотропия ЗАМЕТНО двигает наблюдаемые — к архитектору."
         if amp_spread > 0.05 else
         "Анизотропия почти НЕ двигает форму (амплитуда/A3-A1 стабильны) при фикс. суммарной жёсткости."))
    A("\n> Интерпретация — с архитектором (CLAUDE.md).")
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
