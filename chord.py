import numpy as np
rng=np.random.default_rng(5); N=500_000
def setting(t): t=np.radians(t); return np.array([np.sin(t),0,np.cos(t)])

# ============ СЕМЕЙСТВО ХОРДЫ: P(s,t) ~ |s*a - t*b|^p (симметрично, без "первого") ============
def chord_pair(alpha,beta,p,n=N):
    a,b=setting(alpha),setting(beta)
    w=[]; br=[(1,1),(1,-1),(-1,1),(-1,-1)]
    W=np.array([np.linalg.norm(s*a-t*b)**p for s,t in br]); W/=W.sum()
    idx=rng.choice(4,size=n,p=W)
    S=np.array([br[i][0] for i in idx]); T=np.array([br[i][1] for i in idx])
    return S,T
def E_chord(al,be,p): S,T=chord_pair(al,be,p,200_000); return np.mean(S*T)

print("=== E(θ) семейства хорды: E_p = (sin^p - cos^p)/(sin^p + cos^p) на θ/2 ===")
print("θ°    |  QM    |  p=1   |  p=2   |  p=3   | p=40(~беск)")
for th in [0,30,60,90,120,150,180]:
    row=f"{th:5} | {-np.cos(np.radians(th)):+.3f} |"
    for p in [1,2,3,40]: row+=f" {E_chord(0,th,p):+.3f} |"
    print(row)
print("\nCHSH:")
for p in [1,2,3,40]:
    S=abs(E_chord(0,45,p)-E_chord(0,135,p)+E_chord(90,45,p)+E_chord(90,135,p))
    print(f"  p={p:2}: {S:.3f}")
# no-signaling для всего семейства
for p in [1,3]:
    S1,T1=chord_pair(0,60,p); S2,T2=chord_pair(140,60,p)
    print(f"  p={p}: P(B+|α=0)={np.mean(T1>0):.4f} vs P(B+|α=140)={np.mean(T2>0):.4f}  (no-signaling на симметричной паре)")

# ============ ПРОГРАММА §6: бассейны с РАВНОМЕРНОЙ мерой ============
print("\n=== Попытка 1 (совместный бассейн): λ равномерна на S², ветка = argmax λ·(s*a - t*b) ===")
def cone_pair(alpha,beta,n=N):
    a,b=setting(alpha),setting(beta)
    lam=rng.normal(size=(n,3)); lam/=np.linalg.norm(lam,axis=1,keepdims=True)
    u,w=a+b,a-b
    du,dw=lam@u,lam@w
    same=np.abs(dw)>np.abs(du)          # ветки (s,s): вектор ±w;  ветки (s,-s): ±u
    return np.where(same,1.,-1.)        # сразу произведение S*T
for th in [30,60,90,120,150]:
    st=cone_pair(0,th)
    print(f"  θ={th:3}°: E={np.mean(st):+.3f}   (пила 2θ/π-1 = {2*np.radians(th)/np.pi-1:+.3f},  QM = {-np.cos(np.radians(th)):+.3f})")

print("\n=== Попытка 2 (Кохен-Спекер, один конец): мера НЕ равномерна, релаксация детерминирована ===")
# ось конца n; микроось скрутки λ с плотностью (n·λ)+ ; зажим a; исход = sign(a·λ)
n_ax=setting(0)
for th in [0,30,60,90,120,180]:
    a=setting(th)
    lam=rng.normal(size=(N,3)); lam/=np.linalg.norm(lam,axis=1,keepdims=True)
    keep=rng.random(N)<np.maximum(lam@n_ax,0)          # rejection: плотность (n·λ)+
    lam=lam[keep]
    P=np.mean(lam@a>0)
    print(f"  θ={th:3}°: P(+)={P:.4f}   cos²(θ/2)={np.cos(np.radians(th)/2)**2:.4f}")

print("\n=== Ловушка Белла: та же КС-механика на ПАРЕ с общей λ (локальная релаксация) ===")
lam=rng.normal(size=(N,3)); lam/=np.linalg.norm(lam,axis=1,keepdims=True)
for th in [45,90,135]:
    A=np.sign(lam@setting(0)); B=-np.sign(lam@setting(th))
    print(f"  θ={th:3}°: E={np.mean(A*B):+.3f}  (QM: {-np.cos(np.radians(th)):+.3f})")

# ============ Что выбирает p=2: вчерашняя причинность ============
print("\n=== Несимметричная связь (q=0.85): условное правило семейства f_p(c)=(1+c)^{p/2}/((1+c)^{p/2}+(1-c)^{p/2}) ===")
q=0.85; zc=1-2*q
def f_p(c,p): 
    A=(1+c)**(p/2); B=(1-c)**(p/2); return A/(A+B)
for p in [1,2,3]:
    P_z = q*f_p(-1.0,p)+(1-q)*f_p(1.0,p)   # Алиса мерила z
    P_x = f_p(zc,p)                         # Алиса мерила x
    print(f"  p={p}: P(Боб +z | z)={P_z:.4f}  P(Боб +z | x)={P_x:.4f}  Δ={abs(P_z-P_x):.4f}"
          + ("   <- телеграф" if abs(P_z-P_x)>1e-6 else "   <- причинность цела"))
