import numpy as np
rng = np.random.default_rng(42)
N = 400_000

def rand_units(n):
    v = rng.normal(size=(n,3)); return v/np.linalg.norm(v,axis=1,keepdims=True)

def setting(theta_deg):
    t = np.radians(theta_deg)
    return np.array([np.sin(t),0,np.cos(t)])

# ---------- МОДЕЛЬ 1: локальная скрытая стрелка ----------
# Пара рождается с общим скрытым вектором lambda; A -> sign(a·λ), B -> -sign(b·λ)
# Каждый конец решает САМ, без связи. Это то, что Белл запрещает.
def local_model(alpha, beta, n=N):
    lam = rand_units(n)
    A = np.sign(lam @ setting(alpha)); B = -np.sign(lam @ setting(beta))
    return A, B

# ---------- МОДЕЛЬ 2: "общая линия" ----------
# Пара = один объект G с осью lambda (симметричное нулевое состояние).
# Первое взаимодействие (прибор A, режим alpha) деформирует G:
#   конец A фиксируется в +alpha или -alpha с вероятностью 1/2 (симметрия G, нет выделенного знака)
# Из-за сохранения нулевой суммы структуры конец B становится сопряжённым: ось -A*alpha.
# Прибор B взаимодействует с уже деформированной связью:
#   вероятность режима +1 = cos^2(угол между осью конца B и beta / 2)
#   (правило перекрытия деформаций — здесь оно ПОСТУЛИРУЕТСЯ, это важно)
def bond_model(alpha, beta, n=N):
    A = np.where(rng.random(n) < 0.5, 1.0, -1.0)          # случайный режим конца A
    a, b = setting(alpha), setting(beta)
    axisB = -A[:,None]*a                                   # сопряжённый конец
    cos_t = axisB @ b
    p_plus = (1+cos_t)/2                                   # cos^2(θ/2)
    B = np.where(rng.random(n) < p_plus, 1.0, -1.0)
    return A, B

def E(A,B): return np.mean(A*B)

print("θ°   |  QM=-cosθ | локальная | общая линия")
for th in [0,22.5,45,67.5,90,120,135,180]:
    A1,B1 = local_model(0,th); A2,B2 = bond_model(0,th)
    print(f"{th:5} | {-np.cos(np.radians(th)):+.3f}    | {E(A1,B1):+.3f}    | {E(A2,B2):+.3f}")

# ---------- CHSH ----------
def chsh(model):
    a,ap,b,bp = 0,90,45,135
    return abs(E(*model(a,b)) - E(*model(a,bp)) + E(*model(ap,b)) + E(*model(ap,bp)))
print(f"\nCHSH (классический предел 2, КМ 2√2≈2.828):")
print(f"  локальная:    {chsh(local_model):.3f}")
print(f"  общая линия:  {chsh(bond_model):.3f}")

# ---------- Сигнальность: зависит ли статистика B от настройки A? ----------
print(f"\nP(B=+1) при разных настройках A (β=60° фиксировано):")
for al in [0,45,90,180]:
    _,B = bond_model(al,60)
    print(f"  α={al:3}° -> P(B=+1) = {np.mean(B>0):.4f}")
