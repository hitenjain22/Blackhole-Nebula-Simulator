import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
# shadow angular radius for static observer at D (Synge): sin a = (3sqrt3/D) sqrt(1-2/D)
for D in (20,50,100,1000):
    s=3*np.sqrt(3)/D*np.sqrt(1-2/D); print('D',D,'shadow ang radius deg',np.degrees(np.arcsin(s)))
# deflection angle for impact parameter b (ray from infinity to infinity)
def defl(b):
    # turning point u0: 1/b^2 - u^2 + 2u^3 = 0 smallest positive root
    u0=brentq(lambda u:1/b**2-u*u+2*u**3,1e-12,1/3)
    # substitution u=u0(1-t^2)
    def I(t):
        u=u0*(1-t*t); du=2*u0*t
        val=1/b**2-u*u+2*u**3
        return du/np.sqrt(val)
    v,_=quad(I,0,1,limit=400,epsabs=1e-13,epsrel=1e-13)
    return 2*v-np.pi
for b in (5.3,6.0,10.0,50.0,1000.0):
    print('b',b,'deflection rad',defl(b),'deg',np.degrees(defl(b)),'weak 4/b',4/b)
# Kerr
def risco(a,pro=True):
    Z1=1+(1-a*a)**(1/3)*((1+a)**(1/3)+(1-a)**(1/3)); Z2=np.sqrt(3*a*a+Z1*Z1)
    return 3+Z2-np.sqrt((3-Z1)*(3+Z1+2*Z2)) if pro else 3+Z2+np.sqrt((3-Z1)*(3+Z1+2*Z2))
def rph(a,pro=True):
    return 2*(1+np.cos(2/3*np.arccos(-a if pro else a)))
def rhor(a): return 1+np.sqrt(1-a*a)
for a in (0,0.5,0.9,0.99):
    print('a',a,'horizon',rhor(a),'isco pro',risco(a,True),'retro',risco(a,False),'photon pro',rph(a,True),'retro',rph(a,False))
def xi(r,a): return (-r**3+3*r*r-a*a*(r+1))/(a*(r-1))
def eta(r,a): return r**3*(4*a*a-r*(r-3)**2)/(a*a*(r-1)**2)
for a in (0.5,0.9,0.99):
    rp_,rr_=rph(a,True),rph(a,False)
    # edge-on observer i=90: alpha=-xi, beta=±sqrt(eta)
    print('a',a,'edge-on shadow x-extent: alpha(pro)=',-xi(rp_,a),'alpha(retro)=',-xi(rr_,a), 'eta at pro',eta(rp_,a))
    # vertical extent: max eta over r in [rp_,rr_]
    rs=np.linspace(rp_,rr_,200001); et=eta(rs,a); k=np.argmax(et)
    print('   max beta',np.sqrt(et[k]),'at alpha',-xi(rs[k],a))
# redshift factors
for r in (6,10,50):
    print('r',r,'static g',np.sqrt(1-2/r),'circular b=0 g',np.sqrt(1-3/r))
# Sod exact (gamma 1.4)
g=1.4
rl,pl,ul,rR,pR,uR=1,1,0,0.125,0.1,0
cl,cr=np.sqrt(g*pl/rl),np.sqrt(g*pR/rR)
def fK(p,rk,pk,ck):
    if p>pk:
        A=2/((g+1)*rk); B=(g-1)/(g+1)*pk
        return (p-pk)*np.sqrt(A/(p+B))
    return 2*ck/(g-1)*((p/pk)**((g-1)/(2*g))-1)
ps=brentq(lambda p:fK(p,rl,pl,cl)+fK(p,rR,pR,cr)+uR-ul,1e-6,10)
us=0.5*(ul+uR)+0.5*(fK(ps,rR,pR,cr)-fK(ps,rl,pl,cl))
rsl=rl*(ps/pl)**(1/g)
rsr=rR*((ps/pR)+(g-1)/(g+1))/((g-1)/(g+1)*ps/pR+1)
S=uR+cr*np.sqrt((g+1)/(2*g)*ps/pR+(g-1)/(2*g))
csl=cl*(ps/pl)**((g-1)/(2*g))
print('Sod p*',ps,'u*',us,'rho*L',rsl,'rho*R',rsr,'shock speed',S,'rarefaction head',-cl,'tail',us-csl)
