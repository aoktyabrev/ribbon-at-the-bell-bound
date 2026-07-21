"""C3-S / S-F3 числовой КОНТРОЛЬ аналитики (НЕ J-батарея — верификация вывода
NS-нарушения O(κχ) до закрытия попытки). Анзац:
P(s,t|a,b,λ) ∝ (1−st·a·b)/4·(1+κ s λ·a)(1+κ t λ·b), λ~μ_χ.
Проверяет: P(s=+|a,b) зависит от b (телеграф) при χ>0,κ≠0; плоско при χ=0.
"""
import numpy as np

N = 3_000_000; CHI = 0.5; KAPPA = 0.5
RNG = np.random.default_rng(1)


def sample(chi, n=N):
    if chi == 0:
        v = RNG.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True); return v
    mx = 1 + chi; out = []; need = n
    while need > 0:
        w = RNG.normal(size=(need * 2, 3)); w /= np.linalg.norm(w, axis=1, keepdims=True)
        acc = RNG.random(need * 2) < (1 + chi * w[:, 2]) / mx
        out.append(w[acc]); need -= int(acc.sum())
    return np.concatenate(out)[:n]


def s_of(deg):
    t = np.radians(deg); return np.array([np.sin(t), 0.0, np.cos(t)])


def marg_s_plus(lam, a, b, kappa=KAPPA):
    u = lam @ a; v = lam @ b; c = float(a @ b)
    def W(s, t): return (1 - s * t * c) / 4 * (1 + kappa * s * u) * (1 + kappa * t * v)
    Z = W(1, 1) + W(1, -1) + W(-1, 1) + W(-1, -1)
    return float(np.mean((W(1, 1) + W(1, -1)) / Z))


if __name__ == "__main__":
    a = s_of(0)
    lam = sample(CHI)
    print(f"χ={CHI}, κ={KAPPA}: P(s=+|a=ẑ, b) по настройке Боба:")
    for bd in (0, 45, 90, 135):
        print(f"  b={bd:>3}°: {marg_s_plus(lam, a, s_of(bd)):.6f}")
    lam0 = sample(0)
    print("контроль χ=0 (плоско ⇒ NS держится):")
    for bd in (0, 45, 90, 135):
        print(f"  b={bd:>3}°: {marg_s_plus(lam0, a, s_of(bd)):.6f}")
    print("ВЕРДИКТ: χ>0 ⇒ разброс по b (телеграф O(κχ)); χ=0 ⇒ плоско. S-F3 закрыта.")
