#!/usr/bin/env python
"""R4c — дожатие меры антиферро-сектора Tw=2π (решение архитектора).

Адаптивная лестница (равный acceptance), свопы каждые 10 шагов, полный свип θ (25 точек),
оба сектора, 2 сида, 2 независимые лестницы. Round-trips + ладдер/сид-независимость.
Фиты формы ПОСЛЕ фиксации сырых весов (масштаб.синглет/хорда/ступень/пила, AIC, bootstrap).
Диагностика меры (c_A,c_B по ветвям) на T_cold. Результат — results/R4c/report.md.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim import analysis, plots
from ribbon_sim.dynamics import build_pt_fns, classify, branch_counts
from ribbon_sim.frames import sector_sample, total_twist, axis
from ribbon_sim.pt import make_pt, geometric_ladder
from ribbon_sim.experiment import setting_vectors

N = 64
THETA_DEG = np.arange(0.0, 180.01, 15.0)          # θ-сетка 15° (13 точек)
MEAS_THETA = [0.0, 60.0, 120.0]                   # θ для диагностики меры
SWAP_EVERY = 10
B_RUN = 1024
N_BLOCKS = 6000                                   # 60k шагов (mean-RT≥3 при R≈30, swap10)
kappa_eq = 1.0; total = 3.0 * kappa_eq * (N - 1)
K_B = total / 2.1; K_T = 0.1 * K_B; LR = 0.5 / (max(K_B, K_T) + 1.0)
OUT = ROOT / "results" / "R4d"; OUT.mkdir(parents=True, exist_ok=True)
rows = []


def cfg_pt():
    return {"lr": LR, "k_e": 0.0, "k_c": 1.0, "spinor": True, "elastic": "cosserat_geo",
            "k_b": K_B, "k_t": K_T, "twist_project": True, "n_twist_corr": 10}


def flush():
    (OUT / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def pilot_acc(betas, tw, T_cold, B=256, n_blocks=200):
    """Средний acceptance по звеньям при данной лестнице (пилот на θ=90)."""
    a, b = setting_vectors(np.radians(90.0)); a = jnp.asarray(a); b = jnp.asarray(b)
    ef, sf = build_pt_fns(cfg_pt(), a, b, tw)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    q0, _ = sector_sample(jax.random.PRNGKey(0), (B, N), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B, len(betas), N, 4))
    _, d = run(jax.random.PRNGKey(1), st0, n_blocks)
    return np.asarray(d["accept_frac"])


def adaptive_ladder(tw, T_cold, T_hot=1.5, target=0.35, max_R=44):
    """Адаптивная лестница равного acceptance: бисекция низких звеньев до min-acc≥target."""
    Ts = list(np.geomspace(T_hot, T_cold, 12))      # старт: 12 геом. ступеней
    for _ in range(30):
        betas = 1.0 / jnp.asarray(sorted(Ts, reverse=True))
        acc = pilot_acc(betas, tw, T_cold)
        if acc.min() >= target or len(Ts) >= max_R:
            break
        j = int(np.argmin(acc))                     # худшее звено
        Ts_sorted = sorted(Ts, reverse=True)
        mid = np.sqrt(Ts_sorted[j] * Ts_sorted[j + 1])
        Ts = Ts_sorted; Ts.insert(j + 1, mid)
    Ts = sorted(Ts, reverse=True)
    return 1.0 / jnp.asarray(Ts), len(Ts), float(acc.min())


def run_point(key, theta, tw, betas, n_blocks, want_measure=False):
    a, b = setting_vectors(np.radians(theta)); a = jnp.asarray(a); b = jnp.asarray(b)
    ef, sf = build_pt_fns(cfg_pt(), a, b, tw)
    run = make_pt(ef, sf, betas, SWAP_EVERY)
    k1, k2 = jax.random.split(key)
    q0, twerr = sector_sample(k1, (B_RUN, N), target=tw)
    st0 = jnp.broadcast_to(q0[:, None], (B_RUN, len(betas), N, 4))
    state, diag = run(k2, st0, n_blocks)
    qc = state[:, -1]                               # холодная реплика
    s, t = classify(qc, a, b); s = np.asarray(s); t = np.asarray(t)
    cnt = np.asarray(branch_counts(jnp.asarray(s), jnp.asarray(t)))
    meas = None
    if want_measure:
        nA = np.asarray(axis(qc[:, 0])); nB = np.asarray(axis(qc[:, -1]))
        meas = {"c_A": nA @ np.asarray(a), "c_B": nB @ np.asarray(b), "s": s, "t": t}
    return cnt, diag, float(twerr), meas


def af(cnt):
    c = cnt.astype(float); return (c[1] + c[2]) / max(c.sum(), 1)


def E_of(cnt):
    c = cnt.astype(float); n = c.sum()
    return (c[0] + c[3] - c[1] - c[2]) / max(n, 1)


def analyze_T(res_T, thr):
    """Из счётчиков ячейки {θ:(cnt,diag,twerr,meas)} извлечь E(θ), A(синглет-фит), A1/A3,
    ΔF(θ)=T·ln[(1-P)/P] и его линейность по cosθ."""
    ths = sorted(res_T.keys())
    E = np.array([E_of(res_T[t][0]) for t in ths])
    P = np.array([af(res_T[t][0]) for t in ths])
    ncnt = np.array([res_T[t][0].sum() for t in ths])
    sigE = np.sqrt(np.maximum(1 - E**2, 1e-4) / ncnt)
    thr_a = np.radians(ths)
    fits = analysis.fit_shapes(thr_a, E, sigE)
    h = analysis.harmonics(thr_a, E)
    return {"ths": np.array(ths), "E": E, "P": P, "A_singlet": fits["singlet"]["A"],
            "best": fits["best"], "A1": h["A1"], "A3": h["A3"], "sigE": sigE,
            "chi_singlet": fits["singlet"]["chi2"], "chi_step": fits["step"]["chi2"]}


def main():
    T_GRID = [3e-3, 1e-3, 3e-4, 1e-4]
    rows.append("# R4d — температурная развязка «ножниц» сектора Tw=2π\n")
    rows.append(f"k_t/k_b=0.1, Tw=2π, свип T_cold={T_GRID}, лестница переадаптируется на каждый T, "
                f"θ-сетка 15° (13 точек), B={B_RUN}, свопы/{SWAP_EVERY} шагов, {N_BLOCKS} блоков, 2 сида.\n")
    rows.append("**Гипотеза (архитектор, tanh/линейный отклик):** (a) A(T)↑ с ↓T; (b) форма→ступень "
                "(|A3/A1|↑); (c) коллапс E на −tanh(c(θ)·β), ΔF T-независим. "
                "**ФАЛЬСИФИКАТОР:** A→1 при A3≈0 (чистый косинус до 1) ⇒ кандидат на КВАНТОВУЮ меру, СТОП.\n")
    t0 = time.time()
    per_T = {}          # T_cold -> analyze_T
    meas_cold = None
    ladder_gaps = {}
    for Tc in T_GRID:
        betas_ad, R_ad, accmin = adaptive_ladder(2 * np.pi, Tc)
        betas_geo = 1.0 / geometric_ladder(1.5, Tc, R_ad)
        print(f"T={Tc}: адапт. лестница R={R_ad} min-acc={accmin:.2f} ({(time.time()-t0)/60:.1f}мин)")
        res_ad = {}     # seed0 adapt для основной кривой
        res_ad1 = {}
        for th in THETA_DEG:
            k = jax.random.PRNGKey(hash(("ad", Tc, 0, round(float(th), 1))) % (2**31))
            wm = (Tc == T_GRID[-1] and float(th) in MEAS_THETA)  # мера на самом холодном
            res_ad[float(th)] = run_point(k, th, 2 * np.pi, betas_ad, N_BLOCKS, wm)
            k1 = jax.random.PRNGKey(hash(("ad1", Tc, 1, round(float(th), 1))) % (2**31))
            res_ad1[float(th)] = run_point(k1, th, 2 * np.pi, betas_ad, N_BLOCKS, False)
        # ладдер-проверка на ключевых точках (геом. лестница)
        lg = 0.0
        for th in [0.0, 90.0, 180.0]:
            kg = jax.random.PRNGKey(hash(("geo", Tc, 0, th)) % (2**31))
            cg = run_point(kg, th, 2 * np.pi, betas_geo, N_BLOCKS, False)[0]
            lg = max(lg, abs(af(cg) - af(res_ad[th][0])))
        ladder_gaps[Tc] = lg
        per_T[Tc] = analyze_T(res_ad, np.radians(THETA_DEG))
        # сид-гэп
        sg = max(abs(af(res_ad[float(th)][0]) - af(res_ad1[float(th)][0])) for th in THETA_DEG)
        per_T[Tc]["seed_gap"] = sg
        per_T[Tc]["ladder_gap"] = lg
        per_T[Tc]["R"] = R_ad
        per_T[Tc]["mean_rt"] = np.mean([float(res_ad[float(th)][1]["mean_roundtrips"]) for th in THETA_DEG])
        if Tc == T_GRID[-1]:
            meas_cold = {t: res_ad[t][3] for t in MEAS_THETA if res_ad[t][3]}
        print(f"  T={Tc}: A={per_T[Tc]['A_singlet']:.3f} A3/A1={per_T[Tc]['A3']/per_T[Tc]['A1']:+.3f} "
              f"best={per_T[Tc]['best']} RT={per_T[Tc]['mean_rt']:.1f} sg={sg:.3f} lg={lg:.3f}")
        flush()

    # --- сводка A(T), A3/A1(T) ---
    rows.append("## Развязка: A(T), форма, эквилибрация\n")
    rows.append("| T_cold | R | mean-RT | A (синглет) | A1 | A3 | A3/A1 | лучшая | сид-гэп | ладдер-гэп |")
    rows.append("|---|---|---|---|---|---|---|---|---|---|")
    for Tc in T_GRID:
        d = per_T[Tc]; a31 = d["A3"] / d["A1"] if abs(d["A1"]) > 1e-6 else float("nan")
        rows.append(f"| {Tc:.0e} | {d['R']} | {d['mean_rt']:.1f} | {d['A_singlet']:.3f} | {d['A1']:+.3f} "
                    f"| {d['A3']:+.3f} | {a31:+.3f} | {d['best']} | {d['seed_gap']:.3f} | {d['ladder_gap']:.3f} |")
    flush()

    # --- (0) ΔF(θ) и линейность, мастер-кривая ---
    rows.append("\n## (0)/(c) ΔF(θ)=T·ln[(1−P)/P]: линейность по cosθ и коллапс на tanh\n")
    rows.append("| T_cold | наклон ΔF/T по cosθ | R²(cosθ) |")
    rows.append("|---|---|---|")
    slopes = {}
    for Tc in T_GRID:
        d = per_T[Tc]; P = np.clip(d["P"], 1e-4, 1 - 1e-4); c = np.cos(np.radians(d["ths"]))
        dF = Tc * np.log((1 - P) / P)
        A_ = np.vstack([np.ones_like(c), c]).T; coef, *_ = np.linalg.lstsq(A_, dF, rcond=None)
        pred = A_ @ coef; R2 = 1 - np.sum((dF - pred)**2) / max(np.sum((dF - dF.mean())**2), 1e-12)
        slopes[Tc] = coef[1] / Tc
        rows.append(f"| {Tc:.0e} | {coef[1]/Tc:+.3f} | {R2:.4f} |")
    rows.append(f"\nНаклон ΔF/(T·cosθ) по T: {[f'{Tc:.0e}:{slopes[Tc]:+.2f}' for Tc in T_GRID]}. "
                "Если наклон ПОСТОЯНЕН (ΔF∝T) ⇒ A фикс (масштаб-инвариант); если РАСТЁТ по |值| "
                "(ΔF интринсик) ⇒ A→1.\n")
    flush()

    # --- график A(T) и E(θ) по T ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for Tc in T_GRID:
        d = per_T[Tc]
        ax1.plot(d["ths"], d["E"], "o-", ms=3, label=f"T={Tc:.0e} (A={d['A_singlet']:.2f})")
    g = np.linspace(0, 180, 200)
    ax1.plot(g, -np.cos(np.radians(g)), "k--", lw=1, alpha=0.5, label="−cosθ")
    ax1.plot(g, -np.sign(np.cos(np.radians(g))), "b:", lw=1, alpha=0.5, label="−ступень")
    ax1.set_xlabel("θ°"); ax1.set_ylabel("E(θ)"); ax1.set_title("E(θ) по T_cold"); ax1.legend(fontsize=7); ax1.grid(alpha=0.2)
    As = [per_T[Tc]["A_singlet"] for Tc in T_GRID]; A31 = [abs(per_T[Tc]["A3"]/per_T[Tc]["A1"]) for Tc in T_GRID]
    ax2.plot([f"{t:.0e}" for t in T_GRID], As, "o-", label="A (амплитуда)")
    ax2.plot([f"{t:.0e}" for t in T_GRID], A31, "s-", label="|A3/A1| (форма)")
    ax2.axhline(1.0, color="r", ls="--", lw=1, alpha=0.5, label="A=1 (квант)")
    ax2.set_xlabel("T_cold"); ax2.set_title("A(T) и |A3/A1|(T)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "A_of_T.png", dpi=120); plt.close(fig)
    rows.append("![A(T) и E(θ,T)](A_of_T.png)\n")

    # --- диагностика меры на самом холодном ---
    if meas_cold:
        measure = {t: {"c_A": m["c_A"], "c_B": m["c_B"], "s": m["s"], "t": m["t"],
                       "tau": np.zeros(N - 1), "Tw0": 2*np.pi, "Tw1": 2*np.pi} for t, m in meas_cold.items()}
        plots.plot_measure_diag(measure, OUT / "measure_coldest.png",
                                title=f"R4d мера Tw=2π при T={T_GRID[-1]:.0e}")
        rows.append(f"## Диагностика меры Tw=2π при T={T_GRID[-1]:.0e} (антиферро, КС-оверлей)\n")
        rows.append("![мера](measure_coldest.png)\n")

    # --- вердикт ---
    A_warm, A_cold = per_T[T_GRID[0]]["A_singlet"], per_T[T_GRID[-1]]["A_singlet"]
    a31_warm = abs(per_T[T_GRID[0]]["A3"]/per_T[T_GRID[0]]["A1"])
    a31_cold = abs(per_T[T_GRID[-1]]["A3"]/per_T[T_GRID[-1]]["A1"])
    A_grows = A_cold > A_warm + 0.03
    form_flattens = a31_cold > a31_warm + 0.03
    falsifier = A_cold > 0.9 and a31_cold < 0.05
    rows.append("## Вердикт по пре-регистрации\n")
    rows.append(f"- **(a) A(T)↑ с ↓T:** A {A_warm:.3f}→{A_cold:.3f} — {'✅ РАСТЁТ' if A_grows else '❌ НЕ растёт (A фикс ⇒ мера масштаб-инвариантна)'}.")
    rows.append(f"- **(b) форма→ступень (|A3/A1|↑):** {a31_warm:.3f}→{a31_cold:.3f} — {'✅ уплощается' if form_flattens else '❌ форма стабильна (остаётся косинус)'}.")
    rows.append(f"- **(c) коллапс ΔF/T:** наклоны {[f'{slopes[Tc]:+.2f}' for Tc in T_GRID]} — "
                f"{'постоянны ⇒ ΔF∝T (масштаб-инвариант)' if max(abs(slopes[Tc]-slopes[T_GRID[0]]) for Tc in T_GRID)<0.15 else 'растут ⇒ ΔF интринсик'}.")
    if falsifier:
        rows.append("\n> 🚨 **ФАЛЬСИФИКАТОР СРАБОТАЛ:** A→1 при A3≈0 (чистый косинус до амплитуды 1). "
                    "Гипотеза tanh МЕРТВА. Кандидат на КВАНТОВУЮ меру. СТОП + ЭСКАЛАЦИЯ архитектору.")
    else:
        rows.append(f"\n> Фальсификатор НЕ сработал (A_cold={A_cold:.2f}, |A3/A1|_cold={a31_cold:.3f}). "
                    "Гипотеза термо-ножниц (tanh) в силе / уточняется по (a)-(c).")
    rows.append(f"\nCHSH-рамка: A_cold={A_cold:.3f} ⇒ CHSH≈{A_cold*2*np.sqrt(2):.2f} "
                f"({'внутри классического предела <2' if A_cold*2*np.sqrt(2)<2 else 'ПРЕВЫШАЕТ 2 — квантовая область!'}).")
    rows.append(f"\n---\nВремя: {(time.time()-t0)/60:.1f} мин.")
    flush()
    print(f"R4d готово: A {A_warm:.3f}→{A_cold:.3f}, |A3/A1| {a31_warm:.3f}→{a31_cold:.3f}, falsifier={falsifier}")


if __name__ == "__main__":
    main()
