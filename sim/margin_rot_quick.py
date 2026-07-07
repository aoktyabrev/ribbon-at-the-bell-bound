#!/usr/bin/env python
"""Быстрый тест поворота κ=0.1 (float32, coarse θ) — сигнал анизотропии float32-сетки.
Минимальные параметры ради быстрого прогона в окне простоя чужого GPU-процесса."""
import os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
import jax, jax.numpy as jnp, numpy as np
from ribbon_sim import analysis
from ribbon_sim.dynamics import branch_counts, build_relaxer, classify
from ribbon_sim.frames import haar_quaternions

TH = np.radians(np.arange(0, 180.001, 15.0))  # 13 точек
B, SEEDS, STEPS = 8192, (0, 1), 8000


def rod(ax, an):
    ax = np.asarray(ax, float); ax /= np.linalg.norm(ax)
    K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(an)*K+(1-np.cos(an))*(K@K)


def cfg(R, name):
    k_e = 6.3
    c = {"N":64,"B":B,"k_e":k_e,"k_c":1.0,"spinor":False,"elastic":"geodesic",
         "T0":0.0,"decay":1.0,"steps":STEPS,"lr":0.5/(k_e+1.0)}
    run = build_relaxer(c)["run"]
    a0 = np.array([0.,0.,1.]); counts=np.zeros((len(SEEDS),len(TH),4),np.int64)
    for si,sd in enumerate(SEEDS):
        base=jax.random.PRNGKey(sd)
        for ti,th in enumerate(TH):
            a=jnp.asarray(R@a0,jnp.float32); b=jnp.asarray(R@np.array([np.sin(th),0,np.cos(th)]),jnp.float32)
            ki,kn=jax.random.split(jax.random.fold_in(base,ti))
            q0=haar_quaternions(ki,(B,64))
            qf,_=run(kn,q0,a,b); s,t=classify(qf,a,b); counts[si,ti]=np.asarray(branch_counts(s,t))
    cs=counts.sum(0); ps,pt=analysis.marginals(cs); n=cs.sum(-1)
    z=np.concatenate([np.abs(ps-.5)/analysis.marginal_sigma(ps,n),np.abs(pt-.5)/analysis.marginal_sigma(pt,n)])
    mz=float(z.max()); zt=analysis._bonferroni_z(len(z)); gp=1-(1-analysis._two_sided_p(mz))**len(z)
    print(f"  {name}: max|z|={mz:.2f} порог={zt:.2f} p={gp:.3f} -> {'НОРМА' if mz<zt else 'СМЕЩЕНИЕ'}")
    return f"| {name} | {mz:.2f} | {zt:.2f} | {gp:.3f} | {'в норме' if mz<zt else 'смещение'} |"


t0=time.time()
rows=[cfg(np.eye(3),"axis-aligned"), cfg(rod([1,1,1],0.7),"rot1(111,.7)"), cfg(rod([1,0,1],1.2),"rot2(101,1.2)")]
md=["# κ=0.1 быстрый тест поворота (float32, 13θ, B=8192, 2 сида)\n",
    "| конфиг | max\\|z\\| маргиналов | порог | глоб. p | вывод |","|---|---|---|---|---|",*rows,
    f"\nВремя {time.time()-t0:.0f} с. Если axis-aligned=смещение, а повороты=в норме — 5σ есть float32-анизотропия сетки."]
(ROOT/"results"/"R1"/"margin_rot.md").write_text("\n".join(md)+"\n","utf-8")
print("готово")
