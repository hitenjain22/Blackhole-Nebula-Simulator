import numpy as np, warnings
warnings.filterwarnings('ignore')
from scipy.integrate import quad
from scipy.optimize import brentq
def setup(rp,ra):
    V2=lambda r,L2:(1-2/r)*(1+L2/r**2)
    L2=brentq(lambda L2: V2(rp,L2)-V2(ra,L2),1,1e5); E2=V2(rp,L2); L=np.sqrt(L2)
    up,ua=1/rp,1/ra; mid,half=(up+ua)/2,(up-ua)/2
    # E2-(1-2u)(1+L2u^2) = 2 L2 (u-ua)(up-u)(u3 - u)... use factorised form to avoid cancellation
    # cubic: 2L2 u^3 - L2 u^2 + 2u + (E2-1) = 0 has roots ua, up, u3 ; u3 = 1/2 - ua - up
    u3=0.5-ua-up
    def psi(th):
        f=lambda t: L/np.sqrt(2*L2*(u3-(mid+half*np.cos(t))))  # since (u-ua)(up-u)=half^2 sin^2 t and du = half sin t dt
        return quad(f,0,th,limit=500,epsabs=1e-14,epsrel=1e-14)[0]
    return L,L2,E2,mid,half,psi
for rp in (15.,25.,40.):
    ra=5000.; L,L2,E2,mid,half,psi=setup(rp,ra)
    prec=2*psi(np.pi)-2*np.pi
    thSI=brentq(lambda th: psi(th)-np.pi,1e-6,np.pi)
    rSI=1/(mid+half*np.cos(thSI))
    rc=(L2+np.sqrt(L2*L2-12*L2))/2
    print('r_p %g: L %.8f eps %.6e precession %.5f deg  r_SI %.2f  r_circ %.5f'%(rp,L,(E2-1)/2,np.degrees(prec),rSI,rc))
