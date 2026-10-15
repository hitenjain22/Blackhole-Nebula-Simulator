import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
G=6.67430e-11;c=2.99792458e8;Msun=1.98847e30;AU=1.495978707e11
M=4.3e6*Msun; tu=G*M/c**3; lu=G*M/c**2
print('== units'); print('tu s %.6f  lu m %.6e  lu AU %.6f  lu km %.6e'%(tu,lu,lu/AU,lu/1e3))
rp,ra=25.0,5000.0
def V2(r,L2): return (1-2/r)*(1+L2/r**2)
L2=brentq(lambda L2: V2(rp,L2)-V2(ra,L2),1,1e4); E2=V2(rp,L2); L=np.sqrt(L2); E=np.sqrt(E2)
eps=(E2-1)/2
print('== orbit L %.11f E %.11f eps %.11e'%(L,E,eps))
rc=(L2+np.sqrt(L2*L2-12*L2))/2; print('r_circ %.7f'%rc)
for r in (6.0,10.0,50.0,rc,1000.0):
    v=(r-2)/(r*np.sqrt(r-3)); H=(4-r)/(2*r*(r-3)); Lc2=r*r/(r-3)
    PGN=2*np.pi*r/v; PS=2*np.pi*r**1.5
    print('circ r %.4f v %.10e H %.12e L2 %.12f  P_GN %.4f P_Schw %.4f  P_GN days %.4f h %.3f'%(r,v,H,Lc2,PGN,PS,PGN*tu/86400,PGN*tu/3600))
print('peri t 396223.627 -> days %.4f ; period 792447.254 -> days %.4f'%(396223.627*tu/86400,792447.254*tu/86400))
# exact orbit shape psi(r) from pericentre
up,ua=1/rp,1/ra
def dpsi_du(u): return L/np.sqrt(E2-(1-2*u)*(1+L2*u*u))
def psi_of_r(r):
    u=1/r
    # integrate from u to up with substitution u = up - (up-u)*s^2 ... use cos substitution between ua and up
    # theta such that u = mid+half cos(theta): theta=0 at up, theta=pi at ua
    mid,half=(up+ua)/2,(up-ua)/2
    th=np.arccos((u-mid)/half)
    f=lambda t: L/np.sqrt(E2-(1-2*(mid+half*np.cos(t)))*(1+L2*(mid+half*np.cos(t))**2))*half*np.sin(t)
    v,_=quad(f,0,th,limit=500,epsabs=1e-13,epsrel=1e-13); return v
psi_apo=psi_of_r(ra*(1-1e-15)); print('psi peri->apo deg %.6f'%np.degrees(psi_apo))
rSI=brentq(lambda r: psi_of_r(r)-np.pi, 100, ra*(1-1e-12)); print('== self-intersection radius (psi=180 deg) %.3f'%rSI)
e=(ra-rp)/(ra+rp); p=2*ra*rp/(ra+rp); dphi=2*psi_apo-2*np.pi
print('Kepler-rotated formula r_SI=(1+e)rp/(1-e cos(dphi/2)) = %.3f'%((1+e)*rp/(1-e*np.cos(dphi/2))))
# GN dynamics: time from pericentre to rSI, radial speed there
def acc(x,v):
    r=np.linalg.norm(x); rd=x@v/r; h2=np.sum(np.cross(x,v)**2)
    return -(1/r**3)*(1-2/r)**2*x + 2*rd/(r*(r-2))*v - 3*h2/r**5*x
f=ra/(ra-2); vt=L/(f*ra)
x0=np.array([ra,0,0.]); v0=np.array([0,vt,0.])
def rhs(t,s): return np.concatenate([s[3:],acc(s[:3],s[3:])])
def ev_peri(t,s): return s[:3]@s[3:]
ev_peri.direction=1
def ev_si(t,s): return np.linalg.norm(s[:3])-rSI
sol=solve_ivp(rhs,[0,8e5],np.concatenate([x0,v0]),rtol=1e-12,atol=1e-12,events=[ev_peri,ev_si],method='DOP853',dense_output=True)
tp=sol.t_events[0][0]; tsi=sol.t_events[1]
print('peri t %.3f ; r=rSI at t'%tp, tsi, ' dt from peri:', tsi-tp)
for t in tsi:
    s=sol.sol(t); x,v=s[:3],s[3:]; r=np.linalg.norm(x)
    print('  at rSI: |v| %.5e v_r %.5e v_t %.5e angle deg %.3f'%(np.linalg.norm(v),x@v/r,np.linalg.norm(np.cross(x,v))/r,np.degrees(np.arctan2(x[1],x[0]))))
np.save('orbit_consts.npy',np.array([L,E,eps,rSI,tp]))
