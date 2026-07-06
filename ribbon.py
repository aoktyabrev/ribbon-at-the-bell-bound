import numpy as np
rng = np.random.default_rng(7)
N = 400_000

# ============ ЧАСТЬ 1: лента = кватернион (точка на S³) ============
# Конфигурация ленты между A и B — элемент SU(2), не SO(3).
# Поворот конца на угол phi вокруг оси z: q = (cos phi/2, 0,0, sin phi/2)
def q_rot_z(phi): return np.array([np.cos(phi/2),0,0,np.sin(phi/2)])
def rotmat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
print("=== Трюк с ремнём (числами) ===")
for phi,name in [(2*np.pi,"360°"),(4*np.pi,"720°")]:
    q=q_rot_z(phi)
    same_orient = np.allclose(rotmat(q),np.eye(3))
    same_config = np.allclose(q,[1,0,0,0])
    print(f"поворот {name}: ориентация конца прежняя: {same_orient}, конфигурация ленты прежняя: {same_config}  (q={np.round(q,3)})")

# Расстояние между конфигурациями ленты при физическом угле θ между осями:
th=np.radians(70)
qa=q_rot_z(0); qb=np.array([np.cos(th/2),0,np.sin(th/2)*0+0,0]); 
# спинор вдоль оси, повернутой на θ: q = (cos θ/2, sin θ/2 * axis)
qb=np.array([np.cos(th/2),0,np.sin(th/2),0])
print(f"\nфизический угол осей: {np.degrees(th):.0f}° -> геодезич. угол на S³: {np.degrees(np.arccos(abs(qa@qb))):.0f}°  (полуугол — из геометрии, не постулат)")

# ============ ЧАСТЬ 2: битва правил ============
# Общая механика та же: конец A случайно фиксируется ±a (симметрия нулевого состояния),
# конец B получает сопряжённую ось -A*a. Разница ТОЛЬКО в правиле p(+|угол между осью и прибором):
#   "полусфера":  конец падает к ближайшему полюсу (детерминизм)
#   "линейное по углу": p = 1 - θ/π  (равномерная мера по дуге)
#   "рычаг":      p = (1+cosθ)/2  = проекция оси на диаметр прибора
#                 (тождественно cos²(θ/2) = квадрат перекрытия на S³)
def rule_hemi(c):   return (c>0).astype(float)+0.5*(c==0)
def rule_arc(c):    return 1-np.arccos(np.clip(c,-1,1))/np.pi
def rule_lever(c):  return (1+c)/2
RULES={"полусфера":rule_hemi,"линейное по дуге":rule_arc,"рычаг (S³-перекрытие)":rule_lever}

def setting(t): t=np.radians(t); return np.array([np.sin(t),0,np.cos(t)])
def run(rule,alpha,beta,n=N):
    A=np.where(rng.random(n)<0.5,1.,-1.)
    c=(-A)*(setting(alpha)@setting(beta))
    B=np.where(rng.random(n)<rule(c),1.,-1.)
    return A,B
def E(rule,a,b): A,B=run(rule,a,b); return np.mean(A*B)

print("\n=== E(θ) ===")
print("θ°    | QM     |"+"|".join(f" {k:^20}" for k in RULES))
for th in [0,30,45,60,90,135,180]:
    row=f"{th:5} | {-np.cos(np.radians(th)):+.3f} |"
    row+="|".join(f" {E(f,0,th):+.3f}{'':14}" for f in RULES.values())
    print(row)

print("\n=== CHSH (локальный предел 2, КМ 2.828) ===")
for name,f in RULES.items():
    S=abs(E(f,0,45)-E(f,0,135)+E(f,90,45)+E(f,90,135))
    print(f"  {name}: {S:.3f}")

print("\n=== Повторяемость (измерил ту же ось дважды -> совпадение?) и no-signaling ===")
for name,f in RULES.items():
    # повторное измерение: ось конца уже ±b, меряем снова b: угол 0 или 180
    rep = f(np.array([1.0]))[0]
    A,B=run(f,0,60); A2,B2=run(f,120,60)
    print(f"  {name}: P(повтор)= {rep:.2f};  P(B=+|α=0)={np.mean(B>0):.3f} vs P(B=+|α=120)={np.mean(B2>0):.3f}")
