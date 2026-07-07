#!/usr/bin/env python
"""Диагностика КРАСНОГО контроля R3: ±-симметрия нарушена (до 127σ) при осевыровненных
зажимах. Гипотеза: усиление float32-смещения в жёсткой длинной релаксации у симметричного
седла θ=0 (тот же класс артефакта, что κ=0.1). Тест: повернуть зажимы прочь от осей —
асимметрия должна исчезнуть. Инкрементальная запись results/R3/r3_symmetry_diag.md."""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_relaxer, classify, branch_counts
from ribbon_sim.frames import haar_quaternions

STEPS, B = 150000, 8192
OUT = ROOT / "results" / "R3" / "r3_symmetry_diag.md"
rows = []


def rod(ax, an):
    ax = np.asarray(ax, float); ax /= np.linalg.norm(ax)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(an)*K+(1-np.cos(an))*(K@K)


def cfg():
    return {"N":64,"B":B,"k_e":0.0,"k_c":1.0,"spinor":False,"elastic":"cosserat",
            "k_b":15.75,"k_t":15.75,"T0":0.0,"decay":1.0,"steps":STEPS,"lr":0.0104}


def run_one(R, th, name):
    run = build_relaxer(cfg())["run"]
    a = jnp.asarray(R @ np.array([0.,0.,1.]), jnp.float32)
    b = jnp.asarray(R @ np.array([np.sin(th),0.,np.cos(th)]), jnp.float32)
    q0 = haar_quaternions(jax.random.PRNGKey(0),(B,64))
    qf,_ = run(jax.random.PRNGKey(1), q0, a, b)
    s,t = classify(qf,a,b); c = np.asarray(branch_counts(s,t)); n=c.sum()
    pp,pm,mp,mm = c/n
    asym = abs(pp-mm)
    ps = pp+pm  # P(s=+)
    row = f"| {name} | {np.degrees(th):.0f} | {pp:.3f}/{mm:.3f} | {asym:.3f} | {ps:.3f} | {(pp+mm-pm-mp):+.3f} |"
    rows.append(row); flush()
    print(f"  {name} θ={np.degrees(th):.0f}: P(pp)/P(mm)={pp:.3f}/{mm:.3f} asym={asym:.3f} P(s+)={ps:.3f}")


def flush():
    hdr=["# R3 красный контроль: тест поворота зажимов (изотропный Коссера, θ=0)\n",
         f"B={B}, steps={STEPS}, 1 сид. Гипотеза: асимметрия ±-симметрии — усиленный float32-"
         "артефакт при осевыровненных зажимах у седла θ=0; поворот его убьёт.\n",
         "| зажимы | θ° | P(pp)/P(mm) | \\|P(pp)−P(mm)\\| | P(s=+) | E |","|---|---|---|---|---|---|"]
    OUT.write_text("\n".join(hdr+rows)+"\n","utf-8")


t0=time.time()
run_one(np.eye(3), 0.0, "axis-aligned")
run_one(rod([1,1,1],0.7), 0.0, "rot1(111,.7)")
run_one(rod([1,0,1],1.2), 0.0, "rot2(101,1.2)")
rows.append(f"\nВремя {time.time()-t0:.0f} с. Если axis-aligned даёт большую |P(pp)−P(mm)| и "
            "P(s+)≠0.5, а повёрнутые — ~0 и ~0.5, красный контроль R3 — усиленный float32-"
            "артефакт (как κ=0.1), а не физика.")
flush(); print("r3_diag готово")
