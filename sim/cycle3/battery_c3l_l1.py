"""C3-L / L1 батарея (prereg 0b10f5d). Фолиационная инвариантность:
совместное P(s,t|a,b) не зависит от порядка разрешения (p=2, внутр. модель).
Модель коллапса (B): s=sign(λ·a) детерминирован на λ; исход другого конца —
Bernoulli(f_2(m_s·b_other)) на условном среднем (стиринг C3-B). numpy N=2e6.
"""
import json
import os
import sys

import numpy as np

SMOKE = "--smoke" in sys.argv
N = 200_000 if SMOKE else 2_000_000
RNG = np.random.default_rng(20260721)
CHI = 0.5
SIG = 1.0 / np.sqrt(N)
RES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase_D", "results")


def f2(c):
    return (1 + c) / 2  # p=2, аффинна


def sample_source(n=N):
    mx = 1 + CHI; out = []; need = n
    while need > 0:
        v = RNG.normal(size=(need * 2, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < (1 + CHI * v[:, 2]) / mx
        out.append(v[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def joint_table(lam, first, second):
    """P(s_first, s_second | first, second) в порядке first→second.
    first: детерминир. s=sign(λ·first); условн. среднее m_s; second: Bernoulli
    f_2(m_s·second). Возвращает 2×2 {(+,+),(+,−),(−,+),(−,−)} по осям (first,second)."""
    proj = lam @ first; pos = proj >= 0
    Pp = pos.mean(); Pm = 1 - Pp
    mp = lam[pos].mean(0); mm = lam[~pos].mean(0)
    qp = f2(mp @ second); qm = f2(mm @ second)   # P(second=+ | first=±)
    # ключ по (знак first, знак second)
    return {"++": Pp * qp, "+-": Pp * (1 - qp), "-+": Pm * qm, "--": Pm * (1 - qm)}


def reorder_ab(tab_ba):
    """tab по осям (b,a) → пере-ключить в оси (a,b) для сравнения с a→b."""
    # tab_ba ключи = (знак b, знак a); нужно (знак a, знак b)
    return {"++": tab_ba["++"], "+-": tab_ba["-+"], "-+": tab_ba["+-"], "--": tab_ba["--"]}


def angle_grid():
    """Пары (a,b) в плоскости x–z, углы от +ẑ; сетка A4-репрезентативная."""
    def s(deg):
        t = np.radians(deg); return np.array([np.sin(t), 0.0, np.cos(t)])
    pairs = []
    for da in (0, 30, 45, 60, 90):
        for db in (0, 45, 90, 135):
            pairs.append((da, db, s(da), s(db)))
    return pairs


def L1():
    print("=" * 72); print("L1 — фолиационная инвариантность совместного P(s,t|a,b), p=2"); print("=" * 72)
    lam = sample_source()
    max_joint = 0.0; worst = None; rows = []
    for da, db, a, b in angle_grid():
        tab_ab = joint_table(lam, a, b)                 # оси (a,b)
        tab_ba = reorder_ab(joint_table(lam, b, a))     # оси (b,a)→(a,b)
        d = max(abs(tab_ab[k] - tab_ba[k]) for k in tab_ab)
        rows.append(dict(a=da, b=db, disc=float(d),
                         ab={k: float(v) for k, v in tab_ab.items()},
                         ba={k: float(v) for k, v in tab_ba.items()}))
        if d > max_joint:
            max_joint = d; worst = (da, db)
    print(f"  сетка {len(rows)} пар (a,b); порядки a→b vs b→a на совместном:")
    print(f"  max |P_ab − P_ba| = {max_joint:.6f} при (a,b)={worst}°  (2σ={2*SIG:.4f})")
    # производные: маргиналы Боба по настройке Алисы (no-signaling, должно ≈0)
    a0 = np.array([np.sin(np.radians(30)), 0, np.cos(np.radians(30))])
    b0 = np.array([0.0, 0, 1.0])
    mA_dep = abs(sum(v for k, v in joint_table(lam, a0, b0).items() if k[1] == "+")
                 - sum(v for k, v in joint_table(lam, np.array([1.0, 0, 0]), b0).items() if k[1] == "+"))
    # репитабельность a→b→a: s2==s1?
    proj = lam @ a0; s1 = np.sign(proj); pos = proj >= 0
    # Боб-измерение не перепартиционирует ось a ⇒ s2=sign(λ·a)=s1 тождественно
    rep = float(np.mean(np.sign(lam @ a0) == s1))
    passed = max_joint < 2 * SIG
    print(f"  Боб-маргинал по настройке Алисы (no-signaling): {mA_dep:.6f} (<2σ: {mA_dep<2*SIG})")
    print(f"  репитабельность a→b→a (s2=s1): {rep:.4f}")
    print(f"  L1 ВЕРДИКТ: {'ПРОХОД (совместное порядко-инвариантно <2σ)' if passed else 'СТОП — порядко-зависимость >2σ'}")
    return dict(max_joint_disc=float(max_joint), worst_pair=worst, bob_marginal_dep=float(mA_dep),
                repeatability=rep, rows=rows, sigma=SIG, N=N, passed=bool(passed))


def main():
    print(f"[{'SMOKE' if SMOKE else 'FULL'}] N={N}, χ={CHI}, σ={SIG:.2e}\n")
    r = L1()
    tag = "_smoke" if SMOKE else ""
    json.dump(r, open(os.path.join(RES, f"C3L_L1{tag}.json"), "w"), indent=2)
    print(f"\n  → {RES}/C3L_L1{tag}.json")


if __name__ == "__main__":
    main()
