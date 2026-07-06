"""Свип θ, счётчики ветвей, сохранение результата и отчёт (SPEC §3, §4)."""

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
    """Оси зажимов: a = e_z (θ=0), b под физическим углом θ в плоскости x–z.

    Согласовано с ribbon_model_note (chord.py): a·b = cos θ.
    """
    a = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    b = jnp.array([np.sin(theta), 0.0, np.cos(theta)], dtype=jnp.float32)
    return a, b


def _resolve_mode(cfg, mode):
    """Собирает эффективные параметры прогона для 'full' или 'smoke'."""
    phys, dyn, run = cfg["physics"], cfg["dynamics"], cfg["run"]
    params = {
        "N": int(phys["N"]),
        "k_e": float(phys["k_e"]),
        "k_c": float(phys["k_c"]),
        "spinor": bool(phys.get("spinor", False)),
        "lr": float(dyn["lr"]),
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
        params["B"] = int(sm["B"])
        params["steps"] = int(sm["steps"])
        params["seeds"] = list(sm["seeds"])
        thetas_deg = np.array(sm["theta_deg"], dtype=np.float64)

    params["thetas_deg"] = thetas_deg
    params["thetas"] = np.radians(thetas_deg)
    return params


def run_sweep(cfg, mode="full", seed_override=None):
    """Прогон свипа θ на всех сидах. Возврат dict с counts (n_seed, n_theta, 4)."""
    p = _resolve_mode(cfg, mode)
    if seed_override is not None:
        p["seeds"] = [int(seed_override)]

    relaxer = build_relaxer(p)
    N, B = p["N"], p["B"]
    thetas = p["thetas"]

    all_counts = np.zeros((len(p["seeds"]), len(thetas), 4), dtype=np.int64)
    e_final = np.zeros((len(p["seeds"]), len(thetas)), dtype=np.float64)

    t0 = time.time()
    for si, seed in enumerate(p["seeds"]):
        base_key = jax.random.PRNGKey(int(seed))
        for ti, theta in enumerate(thetas):
            a, b = setting_vectors(theta)
            tk = jax.random.fold_in(base_key, ti)
            k_init, k_noise = jax.random.split(tk)
            q0 = haar_quaternions(k_init, (B, N))
            qf, e_trace = relaxer(k_noise, q0, a, b)
            s, t = classify(qf, a, b)
            cnt = branch_counts(s, t)
            cnt.block_until_ready()
            all_counts[si, ti] = np.asarray(cnt)
            e_final[si, ti] = float(e_trace[-1])
    elapsed = time.time() - t0

    return {
        "params": p,
        "mode": mode,
        "counts": all_counts,
        "e_final": e_final,
        "elapsed_s": elapsed,
    }


def _gpu_mem_mb():
    """Пиковая занятость GPU по jax memory_stats (МБ), если доступно."""
    try:
        dev = jax.devices()[0]
        stats = dev.memory_stats()
        peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use", 0)
        return peak / 1e6
    except Exception:
        return None


def write_report(cfg, res, outdir):
    """results/RX/report.md с кривой, контролями §4.2 и вердиктом."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = res["params"]
    thetas = p["thetas"]
    counts = res["counts"]

    # Счётчики, просуммированные по сидам, для основных наблюдаемых/фитов.
    counts_sum = counts.sum(0)
    E = analysis.E_from_counts(counts_sum)
    ctrl = analysis.check_controls(counts_sum, thetas)

    # Воспроизводимость (если ≥2 сида).
    repro = None
    if counts.shape[0] >= 2:
        repro = analysis.reproducibility(counts[0], counts[1])

    # Сохранение сырых счётчиков.
    np.savez(
        outdir / f"counts_{res['mode']}.npz",
        thetas=thetas,
        thetas_deg=p["thetas_deg"],
        counts=counts,
        e_final=res["e_final"],
    )

    # Графики.
    seeds_E = [analysis.E_from_counts(counts[si]) for si in range(counts.shape[0])]
    curve_png = outdir / f"E_curve_{res['mode']}.png"
    plots.plot_E_curve(thetas, E, curve_png, seeds_E=seeds_E,
                       title=f"{cfg['name']} ({res['mode']}): E(θ)")
    marg_png = outdir / f"marginals_{res['mode']}.png"
    plots.plot_marginals(thetas, ctrl["p_s"], ctrl["p_t"], ctrl["sig_s"],
                         ctrl["sig_t"], marg_png)

    # Гармоники.
    harm = analysis.harmonics(thetas, E)

    # Вердикт R0: E≈0 всюду и маргиналы 0.5.
    max_absE = float(np.max(np.abs(E)))
    # Порог на |E|: 3σ шума при данном N (E≈0 ⇒ σ_E≈1/sqrt(n)).
    n_min = counts_sum.sum(-1).min()
    E_tol = 3.0 / np.sqrt(n_min)
    E_ok = max_absE < E_tol
    verdict_pass = E_ok and ctrl["all_marg_ok"] and ctrl["all_sym_ok"]

    md = _render_md(cfg, res, E, ctrl, repro, harm, max_absE, E_tol,
                    verdict_pass, curve_png.name, marg_png.name)
    (outdir / "report.md").write_text(md, encoding="utf-8")
    return {"verdict_pass": verdict_pass, "max_absE": max_absE, "E_tol": E_tol,
            "ctrl": ctrl, "path": str(outdir / "report.md")}


def _render_md(cfg, res, E, ctrl, repro, harm, max_absE, E_tol, verdict_pass,
               curve_name, marg_name):
    p = res["params"]
    deg = p["thetas_deg"]
    mem = _gpu_mem_mb()
    lines = []
    A = lines.append
    A(f"# {cfg['name']} — отчёт ({res['mode']})\n")
    A(f"**Описание:** {cfg.get('description', '')}\n")
    A(f"**Пре-регистрированное предсказание:** {cfg.get('prediction', '')}\n")
    A("## Конфигурация\n")
    A("```")
    A(f"N={p['N']}  B={p['B']}  steps={p['steps']}  seeds={p['seeds']}")
    A(f"k_e={p['k_e']}  k_c={p['k_c']}  spinor={p['spinor']}")
    A(f"lr={p['lr']}  T0={p['T0']}  decay={p['decay']}")
    A(f"θ-сетка (°): {np.round(deg, 2).tolist()}")
    A("```\n")
    A(f"Время прогона: **{res['elapsed_s']:.1f} с** на {jax.devices()[0].device_kind}."
      + (f" Пик GPU-памяти ≈ {mem:.0f} МБ." if mem else ""))
    A("")
    A(f"![E(θ)]({curve_name})\n")
    A(f"![маргиналы]({marg_name})\n")

    A("## Наблюдаемые\n")
    A("| θ° | n | E(θ) | P(s=+) | P(t=+) | sym_same(σ) | sym_opp(σ) |")
    A("|---|---|---|---|---|---|---|")
    counts_sum = res["counts"].sum(0)
    n = counts_sum.sum(-1)
    for i, th in enumerate(deg):
        A(f"| {th:.1f} | {int(n[i])} | {E[i]:+.4f} | {ctrl['p_s'][i]:.4f} "
          f"| {ctrl['p_t'][i]:.4f} | {ctrl['sym_same_sigma'][i]:.2f} "
          f"| {ctrl['sym_opp_sigma'][i]:.2f} |")
    A("")

    A("## Контроли (SPEC §4.2)\n")
    A(f"- **Маргиналы 0.5 ± 3σ:** {'✅ прошли' if ctrl['all_marg_ok'] else '❌ провал'} "
      f"(все P(s=+),P(t=+) в пределах 3σ от 0.5).")
    A(f"- **±-симметрия P(s,t)=P(−s,−t):** "
      f"{'✅ прошла' if ctrl['all_sym_ok'] else '❌ провал'} "
      f"(max |Δ| = {max(ctrl['sym_same_sigma'].max(), ctrl['sym_opp_sigma'].max()):.2f}σ).")
    if repro is not None:
        repro_ok = repro["max_dE_sigma"] < 3.0
        A(f"- **Воспроизводимость на 2 сидах:** "
          f"{'✅' if repro_ok else '❌'} max|ΔE| = {repro['max_abs_dE']:.4f} "
          f"({repro['max_dE_sigma']:.2f}σ).")
    else:
        A("- **Воспроизводимость:** один сид в этом режиме (см. полный прогон).")
    A("")
    A(f"**Гармоники E(θ):** c0={harm['c0']:+.4f}, A1(cosθ)={harm['A1']:+.4f}, "
      f"A3(cos3θ)={harm['A3']:+.4f}.\n")

    A("## Вердикт\n")
    A(f"Предсказание R0: E(θ) ≈ 0 всюду, маргиналы 0.5.")
    A(f"- max|E(θ)| = **{max_absE:.4f}** при пороге 3σ = {E_tol:.4f} → "
      f"{'в пределах шума ✅' if max_absE < E_tol else 'ПРЕВЫШЕН ❌'}.")
    if verdict_pass:
        A("\n> **R0 ПРОЙДЕН.** Разрезанная лента (k_e=0) даёт нулевую корреляцию "
          "и симметричные маргиналы — классификатор ветвей и зажимы исправны.")
    else:
        A("\n> **R0 ПРОВАЛЕН — СТОП (SPEC §3, R0).** При отсутствии связи корреляция "
          "или маргиналы отклонились от нуля/0.5: баг в зажиме, классификации или динамике.")
    return "\n".join(lines) + "\n"
