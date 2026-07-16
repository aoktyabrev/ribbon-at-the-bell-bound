"""C2-A анализ ПОСЛЕ коммита сырья. Ветвление (а)/(б)/(в) по prereg 2535c8b:
бэкенд-систематика vs seed-scatter занижен vs эталон 0.418 — верхняя флуктуация.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RES = os.path.join(os.path.dirname(HERE), "phase_D", "results")
S_SEED_CANON = 0.024      # цикл-1 seed-audit
A_REF_S1 = 0.418          # S1 эталон k_f×1
A_REF_D2EXT = 0.363       # D2-ext эталон (тот же номинал)


def main():
    g = json.load(open(os.path.join(RES, "C2A_raw_gpu.json")))["cells"]
    c = json.load(open(os.path.join(RES, "C2A_raw_cpu.json")))["cells"]
    gk1, ck1 = g["kf1.0"], c["kf1.0"]
    mu_g, s_g = gk1["mean"], gk1["s_seed"]
    mu_c, s_c = ck1["mean"], ck1["s_seed"]
    diff = abs(mu_g - mu_c)
    sig_diff = float(np.sqrt(s_g**2/3 + s_c**2/3))

    # пул всех k_f×1 A-значений (GPU+CPU) для s_seed
    pool = list(gk1["A"]) + list(ck1["A"])
    s_pool = float(np.std(pool, ddof=1))
    mu_pool = float(np.mean(pool))

    print("=== C2-A: разрешение сдвига A(k_f×1) ===")
    print(f"  GPU k_f×1: μ={mu_g:.4f} s_seed={s_g:.4f} (A={[round(x,4) for x in gk1['A']]})")
    print(f"  CPU k_f×1: μ={mu_c:.4f} s_seed={s_c:.4f} (A={[round(x,4) for x in ck1['A']]})")
    print(f"  |μ_GPU−μ_CPU| = {diff:.4f}  vs 3σ_diff = {3*sig_diff:.4f}  (σ_diff={sig_diff:.4f})")
    print(f"  пул s_seed (GPU+CPU, n={len(pool)}) = {s_pool:.4f}  vs 2×0.024 = {2*S_SEED_CANON}")
    print(f"  пул μ = {mu_pool:.4f}; эталоны: S1={A_REF_S1}, D2-ext={A_REF_D2EXT}")

    # ветвление
    if diff > 3*sig_diff:
        branch = "(а) БЭКЕНД-СИСТЕМАТИКА — стоп-аудит (RNG/редукции/порядок) до науки на GPU"
    elif s_pool > 2*S_SEED_CANON:
        branch = "(б) seed-scatter цикла 1 занижен — пересчёт σ_comb всех сверок, канон-заметка"
    else:
        branch = ("(в) эталон 0.418 — ВЕРХНЯЯ флуктуация; диагностические сверки → "
                  f"объединённое среднее μ_pool={mu_pool:.3f}")
    # доп. диагностика: где 0.418 и 0.363 относительно пула
    dev_s1 = (A_REF_S1 - mu_pool) / s_pool
    dev_d2 = (A_REF_D2EXT - mu_pool) / s_pool
    print(f"\n  ⇒ ВЕТКА {branch}")
    print(f"     0.418 = {dev_s1:+.1f}σ_pool от μ_pool; 0.363 = {dev_d2:+.1f}σ_pool (D2-ext совпадает с пулом)")

    out = dict(prereg="2535c8b",
               gpu=dict(mean=mu_g, s_seed=s_g, A=gk1["A"]),
               cpu=dict(mean=mu_c, s_seed=s_c, A=ck1["A"]),
               diff=diff, sigma_diff=sig_diff, s_pool=s_pool, mu_pool=mu_pool,
               branch=branch, dev_S1_sigma=float(dev_s1), dev_D2ext_sigma=float(dev_d2),
               gpu_anchors={k: g[k]["mean"] for k in g})
    json.dump(out, open(os.path.join(RES, "C2A_analysis.json"), "w"), indent=2, ensure_ascii=False)
    print(f"  → {RES}/C2A_analysis.json")


if __name__ == "__main__":
    main()
