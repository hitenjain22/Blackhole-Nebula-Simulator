import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
def phi_inf(Dc,psi):
    # psi: angle between look direction and direction to hole
    u0=1/Dc; b=Dc*np.sin(psi)/np.sqrt(1-2/Dc)
    du0=np.sqrt(max(1/b**2-u0*u0*(1-2*u0),0))
    if psi>np.pi/2: du0=-du0
    def rhs(p,s): return [s[1],3*s[0]**2-s[0]]
    def esc(p,s): return s[0]
    esc.terminal=True; esc.direction=-1
    def cap(p,s): return s[0]-0.5
    cap.terminal=True
    sol=solve_ivp(rhs,[0,40],[u0,du0],events=[esc,cap],rtol=1e-12,atol=1e-14,method='DOP853')
    if sol.t_events[1].size: return np.nan
    return sol.t_events[0][0]
for Dc in (20.0,50.0):
    psh=np.arcsin(3*np.sqrt(3)/Dc*np.sqrt(1-2/Dc))
    for target,name in ((np.pi,'primary ring'),(3*np.pi,'secondary ring')):
        f=lambda psi: phi_inf(Dc,psi)-target
        lo=psh*1.0000001; hi=np.pi/2*0.999
        # scan
        ps=np.linspace(lo,hi,4000); vals=[f(p) for p in ps]
        root=None
        for i in range(len(ps)-1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1]<0:
                root=brentq(f,ps[i],ps[i+1],xtol=1e-14); break
        print('D',Dc,name,'psi deg',np.degrees(root) if root else None,'shadow deg',np.degrees(psh))
    # escape direction for a few psi
    for psi_deg in (10,20,45,80):
        print('  D',Dc,'psi',psi_deg,'phi_inf deg',np.degrees(phi_inf(Dc,np.radians(psi_deg))))
