import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
def isco(a,pro=True):
    Z1=1+(1-a*a)**(1/3)*((1+a)**(1/3)+(1-a)**(1/3)); Z2=np.sqrt(3*a*a+Z1*Z1)
    return 3+Z2-np.sqrt((3-Z1)*(3+Z1+2*Z2)) if pro else 3+Z2+np.sqrt((3-Z1)*(3+Z1+2*Z2))
def rph(a,pro=True): return 2*(1+np.cos(2/3*np.arccos(-a if pro else a)))
print('== Kerr basics (M=1)')
for a in (0,0.5,0.9,0.99):
    print('a %.2f horizon %.8f ISCO pro %.8f retro %.8f photon pro %.8f retro %.8f'%(a,1+np.sqrt(1-a*a),isco(a),isco(a,False),rph(a),rph(a,False)))
a=0.9
def xi(r): return (-r**3+3*r*r-a*a*(r+1))/(a*(r-1))
def eta(r): return r**3*(4*a*a-r*(r-3)**2)/(a*a*(r-1)**2)
r1,r2=rph(a),rph(a,False)
print('shadow a=0.9 edge-on (observer at infinity, alpha=-xi, beta=+-sqrt(eta)): alpha from %.6f to %.6f'%(-xi(r1),-xi(r2)))
res=minimize_scalar(lambda r:-eta(r),bounds=(r1+1e-9,r2-1e-9),method='bounded',options={'xatol':1e-12})
print('  max beta %.6f at alpha %.6f (r=%.6f)'%(np.sqrt(eta(res.x)),-xi(res.x),res.x))
for r in (6.0,10.0,50.0):
    print('Kerr prograde circular Omega_phi r=%g a=0.9: %.10e (Schw %.10e)'%(r,1/(r**1.5+a),r**-1.5))
print('== nodal precession, a=0.9, prograde circular')
for r in (25.,50.,100.,500.):
    Op=1/(r**1.5+a); Oth=Op*np.sqrt(1-4*a*r**-1.5+3*a*a/r**2)
    print('r %5g: exact Kerr Omega_nod %.6e | LT only 2a/r^3 %.6e | LT+quad leading %.6e | ratio LT/Kerr %.4f'%(r,Op-Oth,2*a/r**3,2*a/r**3-1.5*a*a*r**-3.5,(2*a/r**3)/(Op-Oth)))
# numerical: GN + LT (+Q), two coupling choices
J=np.array([0,0,a])
def gn_acc(x,v):
    r=np.linalg.norm(x); rd=x@v/r; h2=np.sum(np.cross(x,v)**2)
    return -(1/r**3)*(1-2/r)**2*x + 2*rd/(r*(r-2))*v - 3*h2/r**5*x
def lt_acc(x,v):
    r=np.linalg.norm(x); n=x/r; Hv=(2/r**3)*(J-3*(J@n)*n); return np.cross(v,Hv)
def q_acc(x):
    r=np.linalg.norm(x); z=x[2]
    g=(a*a/2)*((3-15*z*z/r**2)*x/r**5+np.array([0,0,6*z/r**5]))
    return -g
def node_rate(r,incl_deg,mode,withQ,norb):
    i=np.radians(incl_deg); v=(r-2)/(r*np.sqrt(r-3))
    x0=np.array([r,0,0.]); v0=v*np.array([0,np.cos(i),np.sin(i)])
    P=2*np.pi*r/v
    if mode=='accel':
        def rhs(t,s):
            x,vv=s[:3],s[3:]; aa=gn_acc(x,vv)+lt_acc(x,vv)+(q_acc(x) if withQ else 0)
            return np.concatenate([vv,aa])
        s0=np.concatenate([x0,v0])
        getL=lambda s: np.cross(s[:3],s[3:])
    else:  # canonical: H_GN + generalized force on p
        def dHdp(x,p):
            rr=np.linalg.norm(x); A=(rr-2)/rr; C=-2*(rr-2)/rr**4; return A*p+C*(p@x)*x
        def dHdx(x,p):
            rr=np.linalg.norm(x); A1=2/rr**2; C=-2*(rr-2)/rr**4; C1=(6*rr-16)/rr**5; px=p@x
            return (0.5*A1*(p@p)+0.5*C1*px*px)*x/rr + C*px*p + x/rr**3
        rr=r; f=rr/(rr-2); p0=f*v0
        def rhs(t,s):
            x,p=s[:3],s[3:]; vv=dHdp(x,p); F=lt_acc(x,vv)+(q_acc(x) if withQ else 0)
            return np.concatenate([vv,-dHdx(x,p)+F])
        s0=np.concatenate([x0,p0]); getL=lambda s: np.cross(s[:3],s[3:])
    T=norb*P
    sol=solve_ivp(rhs,[0,T],s0,rtol=1e-11,atol=1e-13,method='DOP853',t_eval=np.linspace(0,T,4001))
    Ls=np.array([getL(sol.y[:,k]) for k in range(sol.y.shape[1])])
    node=np.unwrap(np.arctan2(Ls[:,0],-Ls[:,1]))
    slope=np.polyfit(sol.t,node,1)[0]
    return slope
for r,norb in ((50.,40),(500.,40)):
    Op=1/(r**1.5+a); Oth=Op*np.sqrt(1-4*a*r**-1.5+3*a*a/r**2); kerr=Op-Oth
    for mode in ('accel','canon'):
        for wq in (False,True):
            s=node_rate(r,10.0,mode,wq,norb)
            print('r %g incl 10: %s%s nodal rate %.6e  (vs 2a/r^3: %.4f, vs exact Kerr: %.4f)'%(r,mode,'+Q' if wq else '   ',s,s/(2*a/r**3),s/kerr))
# per-orbit node shift for cloud orbit, weak field
rp,ra=25.,5000.; p=2*rp*ra/(rp+ra)
print('weak-field LT node shift per radial orbit (r_p=25,r_a=5000,a=0.9): %.5f rad = %.4f deg'%(4*np.pi*a/p**1.5,np.degrees(4*np.pi*a/p**1.5)))
