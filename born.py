import numpy as np
rng=np.random.default_rng(11); N=600_000

# Семейство правил: f(c) = (1+c)/2 + eps*(c^3 - c)
# eps=0 -> КМ. Нечётность g и повторяемость f(1)=1, f(-1)=0 сохранены для ЛЮБОГО eps.
def f(c,eps): return 0.5*(1+c)+eps*(c**3-c)

def setting(t): t=np.radians(t); return np.array([np.sin(t),0,np.cos(t)])
def pair(alpha,beta,eps,b_first=False,n=N):
    a,b=setting(alpha),setting(beta)
    if b_first: a,b=b,a
    X=np.where(rng.random(n)<0.5,1.,-1.)              # первый конец
    Y=np.where(rng.random(n)<f((-X)*(a@b),eps),1.,-1.) # второй конец
    return (Y,X) if b_first else (X,Y)

print("="*70)
print("БАТАРЕЯ 1: убивает ли согласованность одной пары eps != 0 ?")
print("="*70)
for eps in [0.0,0.2]:
    c=np.linspace(-1,1,2001); ok=(f(c,eps).min()>=0)&(f(c,eps).max()<=1)
    A,B=pair(0,60,eps); A2,B2=pair(140,60,eps)
    Bf,Af=pair(0,60,eps,b_first=True)
    # последовательные измерения у Боба: его ось +-a -> мерит b -> мерит d
    a,b,d=setting(0),setting(50),setting(110)
    ax=np.where(rng.random(N)<0.5,1.,-1.)[:,None]*a
    o1=np.where(rng.random(N)<f(ax@b,eps),1.,-1.)
    o2=np.where(rng.random(N)<f((o1[:,None]*b)@d,eps),1.,-1.)
    print(f"\neps={eps}:")
    print(f"  вероятности в [0,1]:        {ok}")
    print(f"  повторяемость f(1)={f(1.0,eps):.2f}, f(-1)={f(-1.0,eps):.2f}")
    print(f"  no-signaling (1 изм):       P(B+|α=0)={np.mean(B>0):.4f} vs P(B+|α=140)={np.mean(B2>0):.4f}")
    print(f"  no-signaling (цепочка b,d): P(o2+)={np.mean(o2>0):.4f} (не зависит от α — сверим ниже)")
    print(f"  независимость порядка:      E(A первым)={np.mean(A*B):+.4f} vs E(B первым)={np.mean(Af*Bf):+.4f}")
    # CHSH и гармоника 3θ
    def E(al,be): X,Y=pair(al,be,eps,n=300_000); return np.mean(X*Y)
    S=abs(E(0,45)-E(0,135)+E(90,45)+E(90,135))
    th=np.linspace(0,np.pi,73); Ecurve=np.array([E(0,np.degrees(t)) for t in th])
    a1=-2*np.trapezoid(Ecurve*np.cos(th),th)/np.pi   # коэф. при cosθ
    a3=-2*np.trapezoid(Ecurve*np.cos(3*th),th)/np.pi # коэф. при cos3θ
    print(f"  CHSH = {S:.3f}   |   E(θ) = -({a1:.3f})cosθ - ({a3:.3f})cos3θ   [теория: 3θ-ампл. = eps/2]")
