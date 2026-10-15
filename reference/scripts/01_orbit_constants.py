import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
rp,ra=25.0,5000.0
def Veff2(r,L2): return (1-2/r)*(1+L2/r**2)
L2=brentq(lambda L2: Veff2(rp,L2)-Veff2(ra,L2), 1, 1e4)
E2=Veff2(rp,L2); L=np.sqrt(L2); E=np.sqrt(E2)
print('L',L,'L2',L2,'E',E,'E-1',E-1,'eps_GN',(E2-1)/2)
ua,up=1/ra,1/rp
def g(th):
    u=(ua+up)/2+(up-ua)/2*np.cos(th)
    du=(up-ua)/2*np.sin(th)
    return L/np.sqrt(E2-(1-2*u)*(1+L2*u*u))*du
val,err=quad(g,0,np.pi,limit=500,epsabs=1e-13,epsrel=1e-13)
prec=2*val-2*np.pi
print('precession rad',prec,'deg',np.degrees(prec),'weakfield deg',np.degrees(6*np.pi/(rp*ra*2/(rp+ra))))
rr=(L2+np.sqrt(L2*L2-12*L2))/2; print('ring r',rr)
def acc(x,v):
    r=np.linalg.norm(x); rd=x@v/r; h2=np.sum(np.cross(x,v)**2)
    return -(1/r**3)*(1-2/r)**2*x + 2*rd/(r*(r-2))*v - 3*h2/r**5*x
f=ra/(ra-2); vt=L/(f*ra)
print('apocentre GN tangential speed',vt)
x0=np.array([ra,0,0.]); v0=np.array([0,vt,0.])
def rhs(t,s): return np.concatenate([s[3:],acc(s[:3],s[3:])])
def ev(t,s): return s[:3]@s[3:]
sol=solve_ivp(rhs,[0,1.7e6],np.concatenate([x0,v0]),rtol=1e-12,atol=1e-10,events=ev,method='DOP853')
for te,se in zip(sol.t_events[0],sol.y_events[0]):
    print('turning t',te,'r',np.linalg.norm(se[:3]),'phi deg',np.degrees(np.arctan2(se[1],se[0])))
s=sol.y[:, -1]; x,v=s[:3],s[3:]; r=np.linalg.norm(x); ff=r/(r-2); rd=x@v/r
ell=ff*np.linalg.norm(np.cross(x,v)); eps=0.5*ff**2*rd**2+0.5*ff*(v@v-rd**2)-1/r
print('ell',ell,'eps',eps)
