import numpy as np
from scipy.optimize import brentq
exec(open('02_generalized_leapfrog_check.py').read().split("x=np.array([7.")[0])
def glf(x,p,h,it=12):
    ph=p.copy()
    for _ in range(it): ph=p-0.5*h*dHdq(x,ph)
    x1=x.copy()
    for _ in range(it): x1=x+0.5*h*(dHdp(x,ph)+dHdp(x1,ph))
    return x1, ph-0.5*h*dHdq(x1,ph)
rp,ra=10.,100.
V2=lambda r,L2:(1-2/r)*(1+L2/r**2)
L2=brentq(lambda L2: V2(rp,L2)-V2(ra,L2),1,1e4); L=np.sqrt(L2)
f=ra/(ra-2); vt=L/(f*ra)
for dt in (2.0,1.0,0.5):
    x=np.array([ra,0,0.]); p=p_from_v(x,np.array([0,vt,0.])); e0=H(x,p); mx=0
    for i in range(int(4*1200/dt)):
        x,p=glf(x,p,dt); mx=max(mx,abs(H(x,p)-e0)/abs(e0))
    print('eccentric rp=10 ra=100 fixed dt %.2f: max |dH/H| over ~4 orbits %.4e'%(dt,mx))
