import numpy as np
from scipy.optimize import brentq
kB=1.380649e-23; mp=1.67262192369e-27; mu=0.6; sigma=5.670374419e-8; c=2.99792458e8; a=4*sigma/c
print('a_rad %.6e'%a)
gam=5/3
def u_of(rho,T): return 1.5*kB*T/(mu*mp)+a*T**4/rho
def P_of(rho,T): return rho*kB*T/(mu*mp)+a*T**4/3
def T_of(rho,u): return np.exp(brentq(lambda lT: np.log(u_of(rho,np.exp(lT)))-np.log(u),0,40,xtol=1e-14))
def G1(rho,T):
    b=(rho*kB*T/(mu*mp))/P_of(rho,T)
    return b+(4-3*b)**2*(gam-1)/(b+12*(gam-1)*(1-b)), b
print('== EOS test points (SI)')
for rho,T in ((1e-3,1e4),(1e-6,1e6),(1e-9,1e5),(1e-9,1e7),(1.6586853e-6,None)):
    if T is None:
        u=4.415185590e14; T=T_of(rho,u)
    u=u_of(rho,T); P=P_of(rho,T); g1,b=G1(rho,T)
    print('rho %.4e T %.6e -> u %.6e P %.6e beta_gas %.6f Gamma1 %.6f | T from (rho,u) %.6e'%(rho,T,u,P,b,g1,T_of(rho,u)))
print('limits Gamma1(beta=1)=%.6f Gamma1(beta=0)=%.6f'%(1+0*(4-3)**2,(4)**2*(gam-1)/(12*(gam-1))))
# shock temperature of head-on stream collision at rSI
vr=2.79797e-2; ush=0.5*vr**2*c**2
print('stream collision: 1/2 v_r^2 = %.4e c^2 -> gas-only T %.3e K'%(0.5*vr**2,ush*2/3*mu*mp/kB))
for rho in (1e-6,1e-8,1e-10):
    print('   rho %.0e: T with radiation %.4e K'%(rho,T_of(rho,ush)))
print('== GN -> Schwarzschild 4-velocity identity check')
rng=np.random.default_rng(0)
worst=0
for _ in range(1000):
    r=rng.uniform(2.5,1e4); n=rng.normal(size=3); n/=np.linalg.norm(n); x=r*n
    p=rng.normal(size=3)*rng.uniform(0.01,2)
    A=(r-2)/r; C=-2*(r-2)/r**4
    H=0.5*A*(p@p)+0.5*C*(p@x)**2-1/r
    pr=p@x/r; Lsq=np.sum(np.cross(x,p)**2)
    lhs=1+2*H; rhs=(1-2/r)*(1+Lsq/r**2)+(1-2/r)**2*pr**2
    worst=max(worst,abs(lhs-rhs)/abs(rhs))
print('max rel |(1+2H) - [(1-2/r)(1+L^2/r^2)+(1-2/r)^2 p_r^2]| = %.2e'%worst)
print('== redshift g = sqrt(1-2/r)/sqrt(1-2/r_cam) / (gamma (1 - beta.n))  [static-frame form]')
def g_circ(r,sign,rcam=np.inf):
    E=(1-2/r)/np.sqrt(1-3/r); L=r/np.sqrt(r-3)
    gam_=E/np.sqrt(1-2/r); beta=(L/r)/gam_
    cam=1.0 if np.isinf(rcam) else np.sqrt(1-2/rcam)
    return np.sqrt(1-2/r)/cam/(gam_*(1-sign*beta)), gam_, beta
for r in (6.0,10.0,50.8896237):
    gp,gm_,b=g_circ(r,1)[0],g_circ(r,-1)[0],g_circ(r,1)[2]
    g0=g_circ(r,0)[0]
    print('r %.4f: local orbital speed beta %.8f gamma %.8f | g approaching %.8f receding %.8f transverse(n perp v) %.8f static %.8f'%(r,b,g_circ(r,1)[1],gp,gm_,g0,np.sqrt(1-2/r)))
# cross check with the b-form for r=10
r=10; ut=1/np.sqrt(1-3/r); Om=r**-1.5; bmax=r/np.sqrt(1-2/r)
print('cross-check r=10 b-form:',[1/(ut*(1-Om*b)) for b in (bmax,-bmax,0)])
print('camera at r_cam=50, static emitter at r=10: g=%.8f'%(np.sqrt(1-2/10)/np.sqrt(1-2/50)))
print('== Wien peaks nm', {T: round(2.897771955e-3/T*1e9,3) for T in (3000,1e4,3e4,1e5,1e6)})
