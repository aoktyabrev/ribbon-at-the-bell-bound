#!/usr/bin/env python
"""R3 красный контроль: чинит ли ОТЖИГ ±-симметрию? Гипотеза: асимметрия при θ=0 —
усиление float32-смещения T=0-спуском через нестабильное седло; шум (отжиг) вернёт 50/50.
Изотропный Коссера, θ=0 и 45, T=0 vs T0=1 отжиг. results/R3/r3_anneal_test.md."""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_relaxer, classify, branch_counts
from ribbon_sim.frames import haar_quaternions
from ribbon_sim.experiment import setting_vectors

B = 8192
OUT = ROOT / "results" / "R3" / "r3_anneal_test.md"
rows = []


def cfg(T0, decay, steps):
    return {"N":64,"B":B,"k_e":0.0,"k_c":1.0,"spinor":False,"elastic":"cosserat",
            "k_b":15.75,"k_t":15.75,"T0":T0,"decay":decay,"steps":steps,"lr":0.0104}


def run_one(name, T0, decay, steps, th_deg):
    run = build_relaxer(cfg(T0,decay,steps))["run"]
    a,b = setting_vectors(np.radians(th_deg))
    accum = np.zeros(4)
    for sd in (0,1):
        q0 = haar_quaternions(jax.random.PRNGKey(sd),(B,64))
        qf,_ = run(jax.random.PRNGKey(100+sd), q0, a, b)
        s,t = classify(qf,a,b); accum += np.asarray(branch_counts(s,t))
    n=accum.sum(); pp,pm,mp,mm = accum/n
    rows.append(f"| {name} | {th_deg:.0f} | {pp:.3f}/{mm:.3f} | {abs(pp-mm):.3f} | {pp+pm:.3f} | {(pp+mm-pm-mp):+.3f} |")
    flush(); print(f"  {name} θ={th_deg}: P(pp)/P(mm)={pp:.3f}/{mm:.3f} asym={abs(pp-mm):.3f} P(s+)={pp+pm:.3f}")


def flush():
    hdr=["# R3 красный контроль: чинит ли отжиг ±-симметрию? (изотропный Коссера)\n",
         f"B={B}, 2 сида. T=0-спуск (150k) vs отжиг T0=1 decay=0.9995 (150k).\n",
         "| режим | θ° | P(pp)/P(mm) | \\|P(pp)−P(mm)\\| | P(s=+) | E |","|---|---|---|---|---|---|"]
    OUT.write_text("\n".join(hdr+rows)+"\n","utf-8")


t0=time.time()
for th in (0.0, 45.0):
    run_one("T=0 GD", 0.0, 1.0, 150000, th)
    run_one("отжиг T0=1", 1.0, 0.9995, 150000, th)
rows.append(f"\nВремя {time.time()-t0:.0f} с. Если отжиг даёт |P(pp)−P(mm)|~0 и P(s+)~0.5, "
            "а T=0 — большую асимметрию, то красный контроль R3 лечится отжигом (шум "
            "симметризует нестабильное седло θ=0); T=0-GD для жёсткого Коссера непригоден.")
flush(); print("готово")
