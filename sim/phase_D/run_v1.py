"""V1 (аннигиляция, D-H1) + контроль V1-C (SO(2)) + обнаруженный нулевой frozen-w.

Приготовление: локальный Tw=4π в середине; релаксация при T>0.
Ветки:
  R4       — S³-оснащение, w свободна (полный R⁴): ожидается аннигиляция, parity=+1.
  frozenW  — S³-оснащение, w заморожена (позиции в срезе): ОБНАРУЖЕННЫЙ нулевой контроль.
  SO2      — контроль V1-C (ADD §8): W=2 держится, аннигиляции нет.
Наблюдаемые: E_twist(t), parity(t) [R4], W(t) [SO2], гистограмма локальной скрутки
(R4-финал vs Tw=0-эталон, KS), времена релаксации, отбраковки. Зеркальные пары.

Запуск:  JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/run_v1.py [--smoke]
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

# --- конфигурация ---
if SMOKE:
    N, BATCH, DT_STEPS, N_BLOCKS, T, WIDTH = 32, 16, 400, 6, 0.04, 0.25
else:
    N, BATCH, DT_STEPS, N_BLOCKS, T, WIDTH = 48, 64, 500, 24, 0.04, 0.25
PARAMS = dict(k_s=20.0, k_b=2.0, k_f=1.0, k_c=1.0, ell=1.0, lr=5e-3)


def run_branch(freeze_w, frame_group, prep, key, W_target=None):
    st = B.build_stepper(PARAMS, freeze_w=freeze_w, frame_group=frame_group)
    if frame_group == "S3":
        return C.relax_blocks_s3(st, prep, T, DT_STEPS, N_BLOCKS, key)
    return C.relax_blocks_so2(st, prep, T, DT_STEPS, N_BLOCKS, key, W_target)


def main():
    key = jr.PRNGKey(20260713)
    tag = "smoke" if SMOKE else "full"
    print(f"[V1 {tag}] N={N} B={BATCH} steps={DT_STEPS}x{N_BLOCKS} T={T}")

    prep4 = C.prep_localized(N, 4 * jnp.pi, BATCH, width=WIDTH)
    W_target = float(B.winding(prep4["phi0"][0]))  # = 2.0
    prof_init = C.twist_profile(prep4["x0"], prep4["u0"])

    k1, k2, k3, k4, k5 = jr.split(key, 5)
    r_r4 = run_branch(False, "S3", prep4, k1)
    r_fz = run_branch(True, "S3", prep4, k2)
    r_so2 = run_branch(True, "SO2", prep4, k3, W_target=W_target)

    # Tw=0-эталон (та же R4-релаксация из тривиального сектора) для KS
    prep0 = C.prep_uniform(N, 0.0, BATCH)
    r_ref = run_branch(False, "S3", prep0, k4)

    # зеркальная пара (изометрия) на R4-ветке: ±-контроль наблюдаемой
    prep4m = C.mirror_prep(prep4)
    r_r4m = run_branch(False, "S3", prep4m, k5)

    # профили делокализации ⟨φ_i⟩(позиция)
    prof_r4 = C.twist_profile(r_r4["x"], r_r4["u"])
    prof_so2 = C.twist_profile(r_so2["x"], B.u_of_phi(r_so2["phi"]))

    # --- KS: R4 локальная скрутка (пул по равновесным блокам) vs Tw=0-эталон ---
    ta_r4 = r_r4["ta_pool"]
    ta_ref = r_ref["ta_pool"]
    ks_stat, ks_p = C.ks_two_sample(ta_r4, ta_ref)

    # --- времена релаксации (шаги до полураспада твист-энергии) ---
    tau_r4 = C.relaxation_halftime(r_r4["e_twist"], DT_STEPS, floor=r_ref["e_twist"][-1])
    tau_so2 = C.relaxation_halftime(r_so2["e_twist"], DT_STEPS)

    summary = dict(
        tag=tag, N=N, batch=BATCH, dt_steps=DT_STEPS, n_blocks=N_BLOCKS, T=T,
        params=PARAMS, W_target=W_target,
        R4=dict(e_twist=r_r4["e_twist"], parity_pos_frac=r_r4["parity_pos_frac"],
                rejected=r_r4["rejected"], tau_half_steps=tau_r4),
        frozenW=dict(e_twist=r_fz["e_twist"], parity_pos_frac=r_fz["parity_pos_frac"],
                     rejected=r_fz["rejected"]),
        SO2=dict(e_twist=r_so2["e_twist"], winding=r_so2["winding"],
                 rejected=r_so2["rejected"], tau_half_steps=tau_so2),
        Tw0_ref_e_twist=r_ref["e_twist"],
        R4_mirror_e_twist=r_r4m["e_twist"],
        KS_R4_vs_Tw0=dict(statistic=ks_stat, pvalue=ks_p),
        mirror_e_twist_absdiff=abs(r_r4["e_twist"][-1] - r_r4m["e_twist"][-1]),
        profile_init=prof_init.tolist(), profile_R4=prof_r4.tolist(),
        profile_SO2=prof_so2.tolist(),
    )
    with open(os.path.join(RES, f"v1_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- печать ключевого ---
    print(f"  R4      E_twist: {r_r4['e_twist'][0]:.3f} -> {r_r4['e_twist'][-1]:.3f}  "
          f"parity(+1 frac) end={r_r4['parity_pos_frac'][-1]:.3f}  rej={r_r4['rejected']:.0f}")
    print(f"  frozenW E_twist: {r_fz['e_twist'][0]:.3f} -> {r_fz['e_twist'][-1]:.3f}  "
          f"(нулевой контроль: ≈R4)")
    print(f"  SO2     E_twist: {r_so2['e_twist'][0]:.3f} -> {r_so2['e_twist'][-1]:.3f}  "
          f"W: {r_so2['winding'][0]:.4f} -> {r_so2['winding'][-1]:.4f}  rej={r_so2['rejected']:.0f}")
    print(f"  Tw0 ref E_twist: {r_ref['e_twist'][0]:.3f} -> {r_ref['e_twist'][-1]:.3f}")
    print(f"  KS(R4 vs Tw0): stat={ks_stat:.4f} p={ks_p:.3f}  "
          f"(p>0.05 ⇒ неотличимы)")
    print(f"  mirror |ΔE_twist|={summary['mirror_e_twist_absdiff']:.2e}")
    print(f"  τ½: R4={tau_r4} steps, SO2={tau_so2} steps")

    # --- фигуры ---
    steps_axis = np.arange(N_BLOCKS + 1) * DT_STEPS

    plt.figure(figsize=(7, 4.5))
    plt.plot(steps_axis, r_r4["e_twist"], "-o", ms=3, label="R⁴ (S³, w свободна)")
    plt.plot(steps_axis, r_fz["e_twist"], "--s", ms=3, label="frozen-w (нулевой контроль)")
    plt.plot(steps_axis, r_so2["e_twist"], "-^", ms=3, label="SO(2)-контроль (V1-C)")
    plt.plot(steps_axis, r_ref["e_twist"], ":", color="gray", label="Tw=0 эталон")
    plt.xlabel("шаги ланжевена"); plt.ylabel("E_twist (k_f=1)")
    plt.title(f"V1: аннигиляция локального 4π (N={N}, B={BATCH}, T={T})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v1_etwist_{tag}.png"), dpi=130); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(steps_axis, r_so2["winding"], "-^", color="C2", label="SO(2): winding W")
    plt.axhline(W_target, color="k", ls=":", label=f"W_target={W_target:.0f}")
    plt.fill_between(steps_axis, W_target - 0.01, W_target + 0.01, alpha=0.15, color="k",
                     label="допуск |W−2|<0.01")
    plt.xlabel("шаги ланжевена"); plt.ylabel("winding W")
    plt.title("V1-C: сохранение winding в SO(2)-ветке (ℤ-твист)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v1_winding_{tag}.png"), dpi=130); plt.close()

    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, max(ta_r4.max(), ta_ref.max()) + 1e-6, 40)
    plt.hist(ta_ref, bins=bins, alpha=0.5, density=True, label="Tw=0 эталон")
    plt.hist(ta_r4, bins=bins, alpha=0.5, density=True, label="R⁴ после релаксации 4π")
    plt.xlabel("локальная скрутка φ_i (рад)"); plt.ylabel("плотность")
    plt.title(f"V1: гистограмма локальной скрутки  KS p={ks_p:.3f}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v1_hist_{tag}.png"), dpi=130); plt.close()

    plt.figure(figsize=(7, 4))
    pos = np.arange(len(prof_init))
    plt.plot(pos, prof_init, "-o", ms=3, color="k", label="начальный (локальный 4π)")
    plt.plot(pos, prof_r4, "-^", ms=3, color="C0", label="R⁴ финал (делокализован)")
    plt.plot(pos, prof_so2, "-s", ms=3, color="C2", label="SO(2) финал (W=2 держится)")
    plt.xlabel("позиция связи i"); plt.ylabel("⟨φ_i⟩ локальная скрутка (рад)")
    plt.title("V1: делокализация твист-солитона (профиль)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"v1_profile_{tag}.png"), dpi=130); plt.close()

    print(f"  фигуры → {FIG}/v1_*_{tag}.png ; сводка → {RES}/v1_{tag}.json")


if __name__ == "__main__":
    main()
