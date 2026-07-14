"""АУДИТ DS2 CHSH (prereg: S>2 → полный аудит до выводов). Точка k_f×4, N=32, M=1200.

Два теста:
(A) ПРЯМОЙ CHSH: 4 корреляции E(a,b) с ПОВОРОТОМ обоих концов (a∈{0,π/2}, b∈{π/4,3π/4}),
    без допущения изотропии. Сверка S_direct с изотропным S=2.58 и проверка изотропии.
(B) ПАССИВНОЕ считывание (крус лупхола): состояние релаксировано с зажимами при
    ФИКСИРОВАННЫХ осях (обе=ê, θ=0); затем те же состояния ЧИТАЮТСЯ проекцией на
    повёрнутые (a,b) БЕЗ зажима во время релаксации — корректный LHV-тест. Если
    S_passive ≤ 2, то S>2 в кампании DS2 — артефакт setting-ЗАВИСИМОГО приготовления
    (зажим-во-время-релаксации), а не подлинная нелокальность.

Запуск: JAX_PLATFORMS=cpu PYTHONPATH=src:phase_D python phase_D/audit_ds2_chsh.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import measurement as M

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
BASE = dict(k_s=20.0, k_b=2.0, k_f=4.0, k_c=2.0, ell=1.0)   # k_f×4
LR, T_MID, N, MREP, STEPS = 5e-3, 0.05, 32, 1200, 4000
A_ANG = [0.0, np.pi/2]           # a-настройки
B_ANG = [np.pi/4, 3*np.pi/4]     # b-настройки


def relax(alpha, gamma, key):
    """Релаксация с зажимами при осях (alpha,gamma). Возврат финальные (x,u)."""
    a, b = M.apparatus_axes_ab(alpha, gamma)
    mini = M.build_minimizer(BASE, lr=LR, freeze_w=False)
    p = M.prep_dynamics(N, True, MREP, key)
    k1, k2 = jr.split(key)
    x, u, _ = mini["run"](k1, p["x0"], p["u0"], p["X0"], p["XL"], a, b, jnp.full((STEPS,), T_MID))
    x, u, _ = mini["run"](k2, x, u, p["X0"], p["XL"], a, b, jnp.full((STEPS // 2,), 0.0))
    return x, u


def corr(u, alpha, gamma):
    """⟨s·t⟩ пассивной проекцией на оси (alpha,gamma) (флип: E→−E)."""
    a, b = M.apparatus_axes_ab(alpha, gamma)
    s, t, degen = M.classify_batch(u, a, b)
    s, t, degen = np.asarray(s), np.asarray(t), np.asarray(degen)
    keep = ~degen
    E = float(np.mean((s * t)[keep]))
    sig = float(np.sqrt(max(1 - E * E, 1e-9) / keep.sum()))
    return -E, sig   # флип


def chsh_from(Es):
    """S = E(a0,b0) − E(a0,b1) + E(a1,b0) + E(a1,b1)."""
    return Es[(0, 0)] - Es[(0, 1)] + Es[(1, 0)] + Es[(1, 1)]


def main():
    key = jr.PRNGKey(20260714 + 303)
    out = {}

    # ===== (A) ПРЯМОЙ CHSH: relax с каждой парой (a,b), читать той же парой =====
    print("=== (A) ПРЯМОЙ CHSH (relax+read одной парой, поворот обоих концов) ===")
    Ed = {}
    for i, al in enumerate(A_ANG):
        for j, ga in enumerate(B_ANG):
            key, sk = jr.split(key)
            x, u = relax(al, ga, sk)
            E, sig = corr(u, al, ga)
            Ed[(i, j)] = E
            print(f"  E(a={al:.3f},b={ga:.3f}) |a−b|={abs(al-ga):.3f}: {E:+.3f}±{sig:.3f}")
    S_direct = chsh_from(Ed)
    print(f"  ⇒ S_direct = {S_direct:+.3f}  |S|={abs(S_direct):.3f}")
    # изотропия: E(0,π/4) vs E(π/2,3π/4) (оба |Δ|=π/4)? и E(π/2,π/4) |Δ|=π/4
    print(f"  изотропия |Δ|=π/4: E(0,π/4)={Ed[(0,0)]:+.3f} E(π/2,π/4)={Ed[(1,0)]:+.3f} "
          f"E(π/2,3π/4)={Ed[(1,1)]:+.3f} (должны совпасть если изотропно)")
    out["direct"] = dict(E={f"{i}{j}": Ed[(i, j)] for i in range(2) for j in range(2)},
                         S=float(S_direct), absS=float(abs(S_direct)))

    # ===== (B) ПАССИВНОЕ считывание: relax при θ=0 (оси=ê,ê), читать проекцией =====
    print("=== (B) ПАССИВНОЕ считывание (relax при ФИКС осях ê,ê; read проекцией) ===")
    key, sk = jr.split(key)
    x0, u0 = relax(0.0, 0.0, sk)   # состояние приготовлено НЕЗАВИСИМО от настроек
    Ep = {}
    for i, al in enumerate(A_ANG):
        for j, ga in enumerate(B_ANG):
            E, sig = corr(u0, al, ga)
            Ep[(i, j)] = E
            print(f"  E_pass(a={al:.3f},b={ga:.3f}): {E:+.3f}±{sig:.3f}")
    S_pass = chsh_from(Ep)
    print(f"  ⇒ S_passive = {S_pass:+.3f}  |S|={abs(S_pass):.3f}")
    out["passive"] = dict(E={f"{i}{j}": Ep[(i, j)] for i in range(2) for j in range(2)},
                          S=float(S_pass), absS=float(abs(S_pass)))

    # ===== вердикт аудита =====
    loophole = abs(S_pass) <= 2.0 and abs(S_direct) > 2.0
    out["verdict"] = (
        "S>2 — АРТЕФАКТ setting-зависимого приготовления (зажим-во-время-релаксации): "
        "пассивное (корректный LHV) считывание даёт |S|≤2, а зажим-во-время даёт |S|>2"
        if loophole else
        "неоднозначно: см. |S_direct| и |S_passive| — требуется дополнительный разбор")
    print(f"  ⇒ ВЕРДИКТ: {out['verdict']}")
    json.dump(out, open(os.path.join(RES, "DS2_audit_chsh.json"), "w"), indent=2)
    print(f"  → {RES}/DS2_audit_chsh.json")


if __name__ == "__main__":
    main()
