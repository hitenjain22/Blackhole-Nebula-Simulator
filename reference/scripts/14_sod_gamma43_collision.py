import numpy as np
from scipy.optimize import brentq
def sod(g,rl=1.0,pl=1.0,rr=0.125,pr=0.1):
    cl=np.sqrt(g*pl/rl); cr=np.sqrt(g*pr/rr)
    def fK(p,rk,pk,ck):
        if p>pk:
            A=2/((g+1)*rk); B=(g-1)/(g+1)*pk; return (p-pk)*np.sqrt(A/(p+B))
        return 2*ck/(g-1)*((p/pk)**((g-1)/(2*g))-1)
    ps=brentq(lambda p: fK(p,rl,pl,cl)+fK(p,rr,pr,cr),1e-6,10,xtol=1e-15)
    us=0.5*(fK(ps,rr,pr,cr)-fK(ps,rl,pl,cl))
    rsl=rl*(ps/pl)**(1/g); rsr=rr*((ps/pr+(g-1)/(g+1))/((g-1)/(g+1)*ps/pr+1))
    S=cr*np.sqrt((g+1)/(2*g)*ps/pr+(g-1)/(2*g)); csl=cl*(ps/pl)**((g-1)/(2*g))
    return ps,us,rsl,rsr,S,-cl,us-csl
for g in (4/3,5/3):
    print('gamma %.4f: p* %.6f u* %.6f rho*L %.6f rho*R %.6f shock %.6f head %.6f tail %.6f'%((g,)+sod(g)))
# strong-shock jump for head-on collision of two cold streams, each speed v toward the centre (lab = CoM frame)
for g in (5/3,4/3):
    v=1.0
    # post-shock gas at rest; shock moves outward at speed vs; RH: rho2/rho1 = (vs+v)/vs; for strong shock rho2/rho1=(g+1)/(g-1)
    comp=(g+1)/(g-1); vs=v/(comp-1)
    print('gamma %.4f strong-shock head-on collision: compression %.4f, shock speed (lab) %.6f v, post-shock u = %.6f v^2 (=v^2/2)'%(g,comp,vs,0.5))
