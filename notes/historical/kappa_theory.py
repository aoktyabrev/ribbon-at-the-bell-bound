import numpy as np
rng=np.random.default_rng(21)

def setting(t): t=np.radians(t); return np.array([np.sin(t),0,np.cos(t)])
def runit(n): v=rng.normal(size=(n,3)); return v/np.linalg.norm(v,axis=1,keepdims=True)

# ---------- 1. ЖЁСТКИЙ ПРЕДЕЛ: один n, спуск E=-[(n·a)²+(n·b)²] ----------
def rigid(theta_deg,B=40_000,steps=600,lr=0.2):
    a,b=setting(0),setting(theta_deg)
    n=runit(B)
    for _ in range(steps):
        g=-2*((n@a)[:,None]*a+(n@b)[:,None]*b)      # градиент E в объемлющем
        g-=np.sum(g*n,axis=1,keepdims=True)*n       # проекция на касательное
        n-=lr*g; n/=np.linalg.norm(n,axis=1,keepdims=True)
    s=np.sign(n@a); t=np.sign(n@b)
    return np.mean(s*t)
print("=== Жёсткий предел (κ→∞): предсказание E=sign(cosθ) ===")
for th in [30,60,85,95,120,150]:
    print(f"  θ={th:3}°: E={rigid(th):+.3f}   sign(cosθ)={np.sign(np.cos(np.radians(th))):+.0f}")

# ---------- 2. ПРИВЕДЁННАЯ МОДЕЛЬ: два конца, интерьер = эфф. пружина ----------
# E = κ·(γ/2)² − (nA·a)² − (nB·b)²,  γ = угол(nA,nB),  k_c=1, k_eff=κ
def reduced(theta_deg,kappa,B=25_000,steps=1200,seed=0):
    r=np.random.default_rng(seed)
    a,b=setting(0),setting(theta_deg)
    nA=r.normal(size=(B,3)); nA/=np.linalg.norm(nA,axis=1,keepdims=True)
    nB=r.normal(size=(B,3)); nB/=np.linalg.norm(nB,axis=1,keepdims=True)
    lr=0.4/(kappa+1.0)
    for _ in range(steps):
        c=np.clip(np.sum(nA*nB,axis=1),-1+1e-7,1-1e-7)
        gam=np.arccos(c); pref=(kappa*gam/2)/np.sqrt(1-c**2)  # d(κγ²/4)/dc * dc/dn
        gA=-pref[:,None]*nB-2*(nA@a)[:,None]*a
        gB=-pref[:,None]*nA-2*(nB@b)[:,None]*b
        gA-=np.sum(gA*nA,axis=1,keepdims=True)*nA
        gB-=np.sum(gB*nB,axis=1,keepdims=True)*nB
        nA-=lr*gA; nB-=lr*gB
        nA/=np.linalg.norm(nA,axis=1,keepdims=True); nB/=np.linalg.norm(nB,axis=1,keepdims=True)
    s=np.sign(nA@a); t=np.sign(nB@b)
    return np.mean(s*t), np.mean(s>0), np.mean(t>0)

thetas=np.arange(0,181,15)
print("\n=== Приведённая модель: E(θ) по κ (ферро-конвенция: E(0)=+1) ===")
print("θ°    | κ=0.1  | κ=1    | κ=10   | ступень | ферро-cos")
curves={}
for k in [0.1,1.0,10.0]:
    curves[k]=[reduced(th,k)[0] for th in thetas]
for i,th in enumerate(thetas):
    print(f"{th:5} | {curves[0.1][i]:+.3f} | {curves[1.0][i]:+.3f} | {curves[10.0][i]:+.3f} |   {np.sign(np.cos(np.radians(th))) if th!=90 else 0:+.0f}    | {np.cos(np.radians(th)):+.3f}")

# контроль маргиналов на одной точке
_,pA,pB=reduced(60,1.0)
print(f"\nконтроль маргиналов (κ=1, θ=60°): P(s+)={pA:.3f}, P(t+)={pB:.3f}")

# ---------- 3. Гармоники и грубая классификация формы ----------
print("\n=== Гармоники A1(cosθ), A3(cos3θ) и амплитуда ===")
tt=np.radians(thetas)
for k in [0.1,1.0,10.0]:
    E=np.array(curves[k])
    A1=2*np.trapezoid(E*np.cos(tt),tt)/np.pi
    A3=2*np.trapezoid(E*np.cos(3*tt),tt)/np.pi
    print(f"  κ={k:4}: амплитуда E(0)={E[0]:+.3f}, A1={A1:+.3f}, A3={A3:+.3f}, A3/A1={A3/A1:+.3f}")
print("  справка: чистый cos: A3/A1=0; ступень: A3/A1=-1/3≈-0.333; пила: A3/A1=+1/9≈+0.111")
np.save('/home/claude/bond/kappa_curves.npy',{'thetas':thetas,**{f'k{k}':curves[k] for k in curves}},allow_pickle=True)
