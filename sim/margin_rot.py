#!/usr/bin/env python
"""Тест поворота для κ=0.1 (по указанию архитектора).

Гипотеза: 5.34σ-смещение маргиналов — анизотропия float32-сетки при ОСЕВЫРОВНЕННЫХ
зажимах (a=e_z, b в плоскости x–z совпадают с координатными осями). Проверка:
глобально повернуть (a,b) прочь от координатных осей (3 ориентации, float32) —
смещение должно сдвинуться/исчезнуть; плюс контрольный прогон float64 (на CPU:
fp64 на consumer-GPU нереалистично медленный — таймбокс архитектора).
Пишем результат ИНКРЕМЕНТАЛЬНО (results/R1/margin_rot.md).
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ribbon_sim import analysis
from ribbon_sim.dynamics import branch_counts, build_relaxer, classify
from ribbon_sim.frames import haar_quaternions

FULL = np.radians(np.arange(0, 180.001, 7.5))       # 25 точек (float32-прогоны)
COARSE = np.radians(np.arange(0, 180.001, 30.0))    # 7 точек (float64/CPU-контроль)
OUT = ROOT / "results" / "R1" / "margin_rot.md"
rows = []


def rodrigues(axis, angle):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def run_config(R, dtype, thetas, B, seeds, steps, device=None):
    k_e = 0.1 * 63.0
    c = {"N": 64, "B": B, "k_e": k_e, "k_c": 1.0, "spinor": False,
         "elastic": "geodesic", "T0": 0.0, "decay": 1.0, "steps": steps,
         "lr": 0.5 / (k_e + 1.0)}
    ctx = jax.default_device(device) if device is not None else _null()
    with ctx:
        run = build_relaxer(c)["run"]
        a_std = np.array([0.0, 0.0, 1.0])
        counts = np.zeros((len(seeds), len(thetas), 4), dtype=np.int64)
        for si, seed in enumerate(seeds):
            base = jax.random.PRNGKey(seed)
            for ti, th in enumerate(thetas):
                a = jnp.asarray(R @ a_std, dtype=dtype)
                b = jnp.asarray(R @ np.array([np.sin(th), 0.0, np.cos(th)]), dtype=dtype)
                k_init, k_noise = jax.random.split(jax.random.fold_in(base, ti))
                q0 = haar_quaternions(k_init, (B, c["N"])).astype(dtype)
                qf, _ = run(k_noise, q0, a, b)
                s, t = classify(qf, a, b)
                counts[si, ti] = np.asarray(branch_counts(s, t))
    csum = counts.sum(0)
    p_s, p_t = analysis.marginals(csum)
    n = csum.sum(-1)
    z = np.concatenate([np.abs(p_s - 0.5) / analysis.marginal_sigma(p_s, n),
                        np.abs(p_t - 0.5) / analysis.marginal_sigma(p_t, n)])
    maxz = float(z.max())
    zthr = analysis._bonferroni_z(len(z))
    gp = 1.0 - (1.0 - analysis._two_sided_p(maxz)) ** len(z)
    return maxz, zthr, gp, float(np.max(np.abs(analysis.E_from_counts(csum))))


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def flush():
    hdr = [
        "# κ=0.1: тест поворота зажимов (float32-анизотропия?)\n",
        f"GPU: {jax.devices()[0].device_kind}. Гипотеза: 5σ-смещение маргиналов — "
        "артефакт float32-сетки при осевыровненных зажимах; поворот (a,b) прочь от осей "
        "его сдвинет/убьёт. float64-контроль — на CPU (fp64 на GPU нереалистичен).\n",
        "| конфигурация | dtype | сетка θ | max\\|z\\| маргиналов | порог | глоб. p | вывод |",
        "|---|---|---|---|---|---|---|",
    ]
    OUT.write_text("\n".join(hdr + rows) + "\n", encoding="utf-8")


def main():
    ROTS = {
        "axis-aligned": np.eye(3),
        "rot1(111,0.7)": rodrigues([1, 1, 1], 0.7),
        "rot2(101,1.2)": rodrigues([1, 0, 1], 1.2),
        "rot3(011,2.0)": rodrigues([0, 1, 1], 2.0),
    }
    t0 = time.time()
    for name, R in ROTS.items():
        mz, zt, gp, _ = run_config(R, jnp.float32, FULL, B=16384, seeds=(0, 1), steps=8000)
        verdict = "в норме (нет 5σ)" if mz < zt else "смещение есть"
        rows.append(f"| {name} | float32 | 25 | {mz:.2f} | {zt:.2f} | {gp:.3f} | {verdict} |")
        flush(); print(f"  {name} f32: max|z|={mz:.2f} p={gp:.3f} [{verdict}]")

    # float64-контроль на CPU, coarse θ
    try:
        cpu = jax.devices("cpu")[0]
        mz, zt, gp, _ = run_config(np.eye(3), jnp.float64, COARSE, B=16384, seeds=(0, 1),
                                   steps=8000, device=cpu)
        verdict = "в норме (нет 5σ)" if mz < zt else "смещение есть"
        rows.append(f"| axis-aligned | **float64/CPU** | 7 | {mz:.2f} | {zt:.2f} | {gp:.3f} | {verdict} |")
        print(f"  axis f64/CPU: max|z|={mz:.2f} p={gp:.3f} [{verdict}]")
    except Exception as e:
        rows.append(f"| axis-aligned | float64/CPU | — | — | — | — | ПРОПУЩЕНО: {type(e).__name__} |")
        print(f"  f64 skipped: {e}")
    flush()

    rows.append("")
    rows.append(f"\nВремя: {time.time()-t0:.1f} с.\n")
    rows.append("**Чтение:** если осевыровненная float32 даёт max|z|>порог, а повёрнутые "
                "и float64 — в норме, 5σ есть артефакт float32-сетки при осевыровненных "
                "зажимах, а не физика.")
    flush()
    print(f"margin_rot → {OUT}")


if __name__ == "__main__":
    main()
