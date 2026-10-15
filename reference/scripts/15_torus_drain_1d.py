import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
tu=21.180252
r0=50.8896237; rin=6.0; rout=5000.0
def run(alpha,hr,rin=6.0,n=600):
    # grid uniform in X=sqrt(r)
    X=np.linspace(np.sqrt(rin),np.sqrt(rout),n); r=X**2
    nu=alpha*hr**2*np.sqrt(r)
    # initial narrow gaussian ring
    w=0.03*r0; S0=np.exp(-0.5*((r-r0)/w)**2); 
    dr=np.gradient(r); m0=np.sum(2*np.pi*r*S0*dr); S0/=m0
    # dSigma/dt = 3/r d/dr [ r^1/2 d/dr (nu Sigma r^1/2) ] ; finite volume on r grid
    rh=0.5*(r[1:]+r[:-1])
    def rhs(t,S):
        S=S.copy(); S[0]=0.0  # zero torque inner boundary: nu*Sigma=0 at rin
        G=nu*S*np.sqrt(r)
        dG=(G[1:]-G[:-1])/(r[1:]-r[:-1])
        F=np.sqrt(rh)*dG  # flux-like
        out=np.zeros_like(S)
        out[1:-1]=3/r[1:-1]*(F[1:]-F[:-1])/(rh[1:]-rh[:-1])
        out[0]=0; out[-1]=0
        return out
    tnu=r0**2/(alpha*hr**2*np.sqrt(r0))
    ts=np.linspace(0,6*tnu,601)
    sol=solve_ivp(rhs,[0,ts[-1]],S0,t_eval=ts,method='BDF',rtol=1e-8,atol=1e-14)
    dr=np.gradient(r)
    M=np.array([np.sum(2*np.pi*r[1:-1]*sol.y[1:-1,k]*dr[1:-1]) for k in range(len(ts))])
    half=np.interp(0.5,-M+1.0,ts) if M[-1]<0.5 else np.nan
    Mdot=-np.gradient(M,ts)
    kpk=np.argmax(Mdot)
    return tnu,half,ts[kpk],Mdot[kpk],M
for alpha,hr in ((0.1,0.3),(0.1,0.1),(0.03,0.3),(0.3,0.3)):
    tnu,half,tpk,mdpk,M=run(alpha,hr)
    Porb=2*np.pi*r0**1.5
    print('alpha %.2f H/r %.2f: t_nu=r^2/nu %.0f (%.1f d, %.1f orbits) | half-mass accreted at t=%.0f (%.1f d, %.1f orbits, %.3f t_nu) | peak Mdot at t=%.0f (%.2f t_nu), peak Mdot*t_nu=%.3f'%(
        alpha,hr,tnu,tnu*tu/86400,tnu/Porb,half,half*tu/86400,half/Porb,half/tnu,tpk,tpk/tnu,mdpk*tnu))
