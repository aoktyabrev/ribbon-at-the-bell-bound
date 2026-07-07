#!/usr/bin/env python
"""R3 мини-fp64 (шаг 3 архитектора): старый atan2-Коссера, N=16, B=256, kt/kb=0.1,
θ∈{0,45,90}, float32 vs float64 (CPU). ПРЕ-РЕГИСТРАЦИЯ: во float64 ±-асимметрия < 3σ
(подтверждение диагноза float32). Если НЕ умирает — диагноз неверен, эскалация."""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent; sys.path.insert(0, str(ROOT / "src"))
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from ribbon_sim.dynamics import build_relaxer, classify, branch_counts
from ribbon_sim.frames import haar_quaternions
from ribbon_sim.experiment import setting_vectors

N, B, STEPS = 16, 256, 40000
total = 3*15/4; kb = total/2.1; kt = 0.1*kb; lr = 0.5/(total+1)  # kt/kb=0.1
THd = [0.0, 45.0, 90.0]
OUT = ROOT / "results" / "R3" / "r3_minifp64.md"; rows = []
cpu = jax.devices("cpu")[0]


def run(dtype, label):
    with jax.default_device(cpu):
        c = {"N":N,"B":B,"k_e":0.0,"k_c":1.0,"spinor":False,"elastic":"cosserat",
             "k_b":kb,"k_t":kt,"T0":0.0,"decay":1.0,"steps":STEPS,"lr":lr}
        rel = build_relaxer(c)["run"]
        worst = 0.0
        for th in THd:
            a,b = setting_vectors(np.radians(th))
            a,b = jnp.asarray(a,dtype), jnp.asarray(b,dtype)
            ac = np.zeros(4)
            for sd in (0,1):
                q0 = haar_quaternions(jax.random.PRNGKey(sd),(B,N)).astype(dtype)
                qf,_ = rel(jax.random.PRNGKey(80+sd), q0, a, b)
                ac += np.asarray(branch_counts(classify(qf,a,b)[0], classify(qf,a,b)[1]))
            n = ac.sum(); pp,pm,mp,mm = ac
            # ±-симметрия P(pp)=P(mm): отклонение в σ
            sig = np.sqrt(max(pp+mm,1))
            z_same = abs(pp-mm)/sig
            z_opp = abs(pm-mp)/np.sqrt(max(pm+mp,1))
            worst = max(worst, z_same, z_opp)
        rows.append(f"| {label} | {worst:.2f} | {'<3σ ✅' if worst<3 else 'НАРУШЕНА ❌'} |")
        flush(); print(f"  {label}: max ±-симм откл = {worst:.2f}σ")


def flush():
    hdr=["# R3 мини-fp64: чинит ли float64 ±-симметрию atan2-Коссера?\n",
         f"N={N}, B={B}, steps={STEPS}, kt/kb=0.1, θ∈{{0,45,90}}, 2 сида (CPU).\n",
         "ПРЕ-РЕГ: float64 → ±-асимметрия <3σ (подтверждение диагноза float32).\n",
         "| dtype | max ±-симм откл (σ) | вердикт |","|---|---|---|"]
    OUT.write_text("\n".join(hdr+rows)+"\n","utf-8")


t0=time.time()
run(jnp.float32, "float32/CPU")
run(jnp.float64, "float64/CPU")
rows.append(f"\nВремя {time.time()-t0:.0f} с. float64 <3σ + float32 >3σ ⇒ красный контроль "
            "R3 подтверждён как float32-артефакт (диагноз верен).")
flush(); print("готово")
