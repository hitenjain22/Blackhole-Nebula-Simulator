import numpy as np
from scipy.integrate import solve_ivp
L,E,eps,rSI,tp=np.load('orbit_consts.npy')
ra=5000.0; R=100.0
def acc(x,v):
    r=np.linalg.norm(x); rd=x@v/r; h2=np.sum(np.cross(x,v)**2)
    return -(1/r**3)*(1-2/r)**2*x + 2*rd/(r*(r-2))*v - 3*h2/r**5*x
f=ra/(ra-2); vt=L/(f*ra)
rng=np.random.default_rng(2)
N=300; pts=[]
while len(pts)<N:
    q=rng.uniform(-1,1,3)
    if q@q<=1: pts.append(q*R)
def rhs(t,s): return np.concatenate([s[3:],acc(s[:3],s[3:])])
def ev_peri(t,s): return s[:3]@s[3:]
ev_peri.direction=1
def ev_si(t,s): return np.linalg.norm(s[:3])-rSI
TP=[];SI=[]
for q in pts:
    x0=np.array([ra,0,0.])+q; v0=np.array([0,vt,0.])
    sol=solve_ivp(rhs,[0,2.45e6],np.concatenate([x0,v0]),rtol=1e-10,atol=1e-10,events=[ev_peri,ev_si],method='DOP853')
    TP.append(sol.t_events[0][:3]); SI.append(sol.t_events[1][:6])
TP=np.array(TP); SI=np.array(SI)
for k in range(3):
    t=TP[:,k]; tin=SI[:,2*k]; tout=SI[:,2*k+1]
    print('passage %d: peri t mean %.0f range %.0f..%.0f (full %.0f, std %.0f) | in@rSI last %.0f, out@rSI first %.0f -> overlap %s'%(
        k+1,t.mean(),t.min(),t.max(),t.max()-t.min(),t.std(),tin.max(),tout.min(), 'YES (%.0f long)'%(tin.max()-tout.min()) if tin.max()>tout.min() else 'no'))
    if tin.max()>tout.min():
        print('   mass fraction of incoming gas still before rSI when outgoing gas first arrives: %.3f'%np.mean(tin>tout.min()))
