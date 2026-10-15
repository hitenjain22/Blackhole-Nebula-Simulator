import numpy as np
from scipy.integrate import solve_ivp
L,E,eps,rSI,tp=np.load('orbit_consts.npy')
ra=5000.0; R=100.0
def acc(x,v):
    r=np.linalg.norm(x); rd=x@v/r; h2=np.sum(np.cross(x,v)**2)
    return -(1/r**3)*(1-2/r)**2*x + 2*rd/(r*(r-2))*v - 3*h2/r**5*x
def H_of(x,v):
    r=np.linalg.norm(x); f=r/(r-2); g=2*r/(r-2)**2; rd=x@v/r
    p=f*v+g*rd*x/r; A=(r-2)/r; C=-2*(r-2)/r**4
    return 0.5*A*(p@p)+0.5*C*(p@x)**2-1/r
f=ra/(ra-2); vt=L/(f*ra)
rng=np.random.default_rng(1)
N=400; pts=[]
while len(pts)<N:
    q=rng.uniform(-1,1,3)
    if q@q<=1: pts.append(q*R)
pts=np.array(pts)
def rhs(t,s): return np.concatenate([s[3:],acc(s[:3],s[3:])])
def ev_peri(t,s): return s[:3]@s[3:]
ev_peri.direction=1
def ev_si_in(t,s): return np.linalg.norm(s[:3])-rSI
tper=[];Hs=[];tsi_in=[];tsi_out=[]
for q in pts:
    x0=np.array([ra,0,0.])+q; v0=np.array([0,vt,0.])
    Hs.append(H_of(x0,v0))
    sol=solve_ivp(rhs,[0,7e5],np.concatenate([x0,v0]),rtol=1e-10,atol=1e-10,events=[ev_peri,ev_si_in],method='DOP853')
    tper.append(sol.t_events[0][0]); ts=sol.t_events[1]
    tsi_in.append(ts[0]); tsi_out.append(ts[1] if len(ts)>1 else np.nan)
tper=np.array(tper);Hs=np.array(Hs);tsi_in=np.array(tsi_in);tsi_out=np.array(tsi_out)
print('H mean %.6e  min %.6e max %.6e  rel spread +- %.4f'%(Hs.mean(),Hs.min(),Hs.max(),(Hs.max()-Hs.min())/2/abs(Hs.mean())))
print('R/ra^2 = %.3e ; half-range of H = %.3e'%(R/ra**2,(Hs.max()-Hs.min())/2))
print('first pericentre time: mean %.0f  min %.0f  max %.0f  std %.0f'%(tper.mean(),tper.min(),tper.max(),tper.std()))
print('percentiles 5,25,50,75,95:',np.percentile(tper,[5,25,50,75,95]).round(0))
# overlap at rSI: incoming arrivals vs outgoing departures
print('incoming at rSI: %.0f .. %.0f ; outgoing at rSI: %.0f .. %.0f'%(tsi_in.min(),tsi_in.max(),np.nanmin(tsi_out),np.nanmax(tsi_out)))
ov0=max(tsi_in.min(),np.nanmin(tsi_out)); ov1=min(tsi_in.max(),np.nanmax(tsi_out))
print('overlap window at rSI: %.0f .. %.0f (%.0f long)'%(ov0,ov1,ov1-ov0))
# fraction of mass that is incoming at rSI after the first outgoing material arrives there
t_first_out=np.nanmin(tsi_out)
print('fraction of cloud still incoming (not yet past rSI) when first outgoing gas reaches rSI: %.3f'%np.mean(tsi_in>t_first_out))
# correlation: which part of the cloud arrives first
for lab,ax in (('x',0),('y',1),('z',2)):
    print('corr(t_peri, %s offset) = %.3f'%(lab,np.corrcoef(tper,pts[:,ax])[0,1]))
# radial period spread
P=lambda H: None
