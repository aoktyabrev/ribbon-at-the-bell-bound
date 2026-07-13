"""V2 (чётность, D-H2): Tw=2π, M реплик — parity=−1 на ВСЕХ принятых траекториях;
доля сингулярных отбраковок / шаг для трёх значений dt → тренд к 0.

Kill (D-H2): отбраковка не падает с dt ⇒ «чётность» — артефакт фильтра.
Зеркальные пары (u→u⊗j) — ±-контроль.

Запуск:  JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_v2.py [--smoke]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import band4d as B
import d0_common as C

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "fig")
RES = os.path.join(HERE, "results")

# T повышена, чтобы термофлуктуации иногда приближали связь к стене лифта ⇒ ненулевая
# отбраковка с ЯВНЫМ трендом →0 при dt→0 (иначе 2π-равномерный далёк от стен, rej≡0).
if SMOKE:
    N, BATCH, DT_STEPS, N_BLOCKS, T = 32, 24, 300, 5, 0.16
    DTS = [3e-3, 5e-3, 8e-3, 1.2e-2]
else:
    N, BATCH, DT_STEPS, N_BLOCKS, T = 48, 48, 500, 16, 0.16
    DTS = [3e-3, 5e-3, 8e-3, 1.2e-2]
K_BASE = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=1.0, ell=1.0)


def run_at_lr(lr, prep, key):
    """Релаксация 2π при данном dt=lr. Возврат: parity_frac(-1) по блокам,
    доля отбраковок/шаг, финальные x,u."""
    params = dict(K_BASE, lr=lr)
    st = B.build_stepper(params, freeze_w=False, frame_group="S3")
    res = C.relax_blocks_s3(st, prep, T, DT_STEPS, N_BLOCKS, key, sample_last=1)
    neg_frac = [1.0 - f for f in res["parity_pos_frac"]]   # доля parity=−1
    rej_per_step = res["rejected"] / (BATCH * res["n_steps"])
    return dict(neg_frac=neg_frac, rej_per_step=rej_per_step,
                min_neg_frac=min(neg_frac), x=res["x"], u=res["u"])


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260713 + 2)
    print(f"[V2 {tag}] N={N} M={BATCH} steps={DT_STEPS}x{N_BLOCKS} T={T} dts={DTS}")

    prep = C.prep_uniform(N, 2 * jnp.pi, BATCH)
    p0 = float((C.parity_all(prep["x0"], prep["u0"], prep["UA"], prep["UB"]) < 0).mean())
    print(f"  init parity=−1 доля: {p0:.3f} (ожид 1.0)")

    results = {}
    rej_trend = []
    for lr in DTS:
        key, sk = jr.split(key)
        r = run_at_lr(lr, prep, sk)
        results[lr] = r
        rej_trend.append(r["rej_per_step"])
        print(f"  dt={lr:.0e}: parity=−1 доля min={r['min_neg_frac']:.3f} "
              f"(все блоки), отбраковка/шаг={r['rej_per_step']:.3e}")

    # зеркальная пара при среднем dt (изометрия: оснащение И реперы-ссылки)
    prep_m = C.mirror_prep(prep)
    key, sk = jr.split(key)
    rm = run_at_lr(DTS[1], prep_m, sk)
    print(f"  зеркало (dt={DTS[1]:.0e}): parity=−1 доля min={rm['min_neg_frac']:.3f}")

    all_minus_one = all(results[lr]["min_neg_frac"] == 1.0 for lr in DTS) and rm["min_neg_frac"] == 1.0
    trend_down = rej_trend[0] < rej_trend[-1]   # меньший dt ⇒ меньше отбраковок
    print(f"  ⇒ parity=−1 держится на ВСЕХ принятых: {all_minus_one}")
    print(f"  ⇒ отбраковка/шаг падает с dt: {trend_down}  {[f'{x:.2e}' for x in rej_trend]}")

    summary = dict(tag=tag, N=N, M=BATCH, T=T, dts=DTS,
                   init_neg_frac=p0,
                   per_dt={f"{lr:.0e}": dict(neg_frac=results[lr]["neg_frac"],
                                             rej_per_step=results[lr]["rej_per_step"],
                                             min_neg_frac=results[lr]["min_neg_frac"])
                           for lr in DTS},
                   mirror_min_neg_frac=rm["min_neg_frac"],
                   all_minus_one=bool(all_minus_one), rej_trend_down=bool(trend_down))
    with open(os.path.join(RES, f"v2_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- фигуры ---
    steps_axis = np.arange(N_BLOCKS + 1) * DT_STEPS
    plt.figure(figsize=(7, 4))
    for lr in DTS:
        plt.plot(steps_axis, results[lr]["neg_frac"], "-o", ms=3, label=f"dt={lr:.0e}")
    plt.ylim(-0.05, 1.15); plt.axhline(1.0, color="k", ls=":", label="parity=−1 (все)")
    plt.xlabel("шаги ланжевена"); plt.ylabel("доля реплик с parity=−1")
    plt.title(f"V2: сохранение чётности −1 (Tw=2π, M={BATCH})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v2_parity_{tag}.png"), dpi=130); plt.close()

    plt.figure(figsize=(6, 4))
    plt.loglog(DTS, rej_trend, "-o")
    plt.xlabel("dt (шаг ланжевена = lr)"); plt.ylabel("сингулярные отбраковки / шаг")
    plt.title("V2: тренд отбраковки → 0 при dt→0 (D-H2)")
    plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v2_rejtrend_{tag}.png"), dpi=130); plt.close()

    print(f"  фигуры → {FIG}/v2_*_{tag}.png ; сводка → {RES}/v2_{tag}.json")


if __name__ == "__main__":
    main()
