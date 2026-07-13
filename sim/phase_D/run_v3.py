"""V3 (термическая недостижимость): старт из ЧЁТНОГО сектора (u≡const, Tw=0),
умеренная T — спонтанных переходов в НЕЧЁТНЫЙ ноль (перенос фазы C: «заряд только
приготовлением» в R⁴).

Наблюдаемые: доля parity=+1 по блокам (должна ≡1), число переходов (=0), доля
попыток пересечь стену лифта (rejected/шаг — «давление» на барьер; ≈0 из чётного
сектора, т.к. равномерный репер далёк от стен). Зеркальные пары.

Запуск:  JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_v3.py [--smoke]
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

if SMOKE:
    N, BATCH, DT_STEPS, N_BLOCKS = 32, 32, 400, 6
    TEMPS = [0.05, 0.10]
else:
    N, BATCH, DT_STEPS, N_BLOCKS = 48, 64, 500, 24
    TEMPS = [0.05, 0.10, 0.15]
PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=1.0, ell=1.0, lr=5e-3)


def run_at_T(T, prep, key):
    st = B.build_stepper(PARAMS, freeze_w=False, frame_group="S3")
    res = C.relax_blocks_s3(st, prep, T, DT_STEPS, N_BLOCKS, key, sample_last=1)
    pos_frac = res["parity_pos_frac"]                 # доля parity=+1 по блокам
    transitions = sum(1 for f in pos_frac if f < 1.0)  # блоки с любым переходом
    rej_per_step = res["rejected"] / (BATCH * res["n_steps"])
    return dict(pos_frac=pos_frac, min_pos_frac=min(pos_frac),
                transitions=transitions, rej_per_step=rej_per_step)


def main():
    tag = "smoke" if SMOKE else "full"
    key = jr.PRNGKey(20260713 + 3)
    print(f"[V3 {tag}] N={N} M={BATCH} steps={DT_STEPS}x{N_BLOCKS} T={TEMPS}")

    # чётный сектор: u≡const (Tw=0). Разнообразие форм позиций для честных реплик.
    prep = C.prep_uniform(N, 0.0, BATCH, sigma=0.05, key=key)
    p0 = float((C.parity_all(prep["x0"], prep["u0"], prep["UA"], prep["UB"]) > 0).mean())
    print(f"  init parity=+1 доля: {p0:.3f} (ожид 1.0)")

    results = {}
    for T in TEMPS:
        key, sk = jr.split(key)
        r = run_at_T(T, prep, sk)
        results[T] = r
        print(f"  T={T:.2f}: parity=+1 доля min={r['min_pos_frac']:.3f}  "
              f"переходов(блоков)={r['transitions']}  отбраковка/шаг={r['rej_per_step']:.3e}")

    # зеркальная пара при средней T (изометрия: оснащение И реперы-ссылки)
    prep_m = C.mirror_prep(prep)
    key, sk = jr.split(key)
    rm = run_at_T(TEMPS[len(TEMPS) // 2], prep_m, sk)
    print(f"  зеркало: parity=+1 доля min={rm['min_pos_frac']:.3f}  переходов={rm['transitions']}")

    zero_transitions = all(results[T]["transitions"] == 0 for T in TEMPS) and rm["transitions"] == 0
    print(f"  ⇒ спонтанных переходов чёт→нечёт: {'НЕТ (0)' if zero_transitions else 'ЕСТЬ!'}")

    summary = dict(tag=tag, N=N, M=BATCH, temps=TEMPS, init_pos_frac=p0,
                   per_T={f"{T:.2f}": dict(min_pos_frac=results[T]["min_pos_frac"],
                                           transitions=results[T]["transitions"],
                                           rej_per_step=results[T]["rej_per_step"],
                                           pos_frac=results[T]["pos_frac"])
                          for T in TEMPS},
                   mirror_transitions=rm["transitions"],
                   zero_transitions=bool(zero_transitions))
    with open(os.path.join(RES, f"v3_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- фигуры ---
    steps_axis = np.arange(N_BLOCKS + 1) * DT_STEPS
    plt.figure(figsize=(7, 4))
    for T in TEMPS:
        plt.plot(steps_axis, results[T]["pos_frac"], "-o", ms=3, label=f"T={T:.2f}")
    plt.ylim(-0.05, 1.15); plt.axhline(1.0, color="k", ls=":", label="parity=+1 (все)")
    plt.xlabel("шаги ланжевена"); plt.ylabel("доля реплик с parity=+1")
    plt.title(f"V3: чётный сектор не переходит в нечётный (Tw=0, M={BATCH})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v3_parity_{tag}.png"), dpi=130); plt.close()

    print(f"  фигуры → {FIG}/v3_*_{tag}.png ; сводка → {RES}/v3_{tag}.json")


if __name__ == "__main__":
    main()
