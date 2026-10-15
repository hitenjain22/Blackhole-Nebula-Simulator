import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
def u_at(D,psi,phi_target):
    u0=1/D; b=D*np.sin(psi)/np.sqrt(1-2/D)
    du0=np.sqrt(max(1/b**2-u0*u0*(1-2*u0),0))
    def rhs(p,s): return [s[1],3*s[0]**2-s[0]]
    def cap(p,s): return s[0]-0.5
    cap.terminal=True
    sol=solve_ivp(rhs,[0,phi_target],[u0,du0],rtol=1e-12,atol=1e-14,method='DOP853',events=[cap])
    if sol.status==1: return np.nan
    return sol.y[0,-1]   # may be negative if escaped (u<0 means beyond infinity)
for D,rs in ((50.0,30.0),(50.0,100.0),(30.0,15.0)):
    psh=np.arcsin(3*np.sqrt(3)/D*np.sqrt(1-2/D))
    out=[]
    for phit in (np.pi,3*np.pi):
        f=lambda psi: u_at(D,psi,phit)-1/rs
        ps=np.linspace(psh*(1+1e-9),np.pi/2*0.999,4000); vals=np.array([f(p) for p in ps])
        root=np.nan
        for i in range(len(ps)-1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1]<0:
                root=np.degrees(brentq(f,ps[i],ps[i+1],xtol=1e-14)); break
        out.append(root)
    print('camera D=%g, point source r=%g directly behind: primary %.6f deg, secondary %.6f deg (shadow %.6f)'%(D,rs,out[0],out[1],np.degrees(psh)))
# sanity: source at infinity limit, D=50 -> 17.872702
f=lambda psi: u_at(50.0,psi,np.pi)-1e-12
print('check source very far, D=50:',np.degrees(brentq(lambda p: u_at(50.0,p,np.pi),np.radians(10),np.radians(40),xtol=1e-14)))
