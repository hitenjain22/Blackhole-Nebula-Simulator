import numpy as np
def p_from_v(x,v):
    r=np.sqrt(x@x); f=r/(r-2); g=2*r/(r-2)**2; rd=x@v/r
    return f*v+g*rd*x/r
def dHdp(x,p):
    r=np.sqrt(x@x); A=(r-2)/r; C=-2*(r-2)/r**4
    return A*p+C*(p@x)*x
def dHdq(x,p):
    r=np.sqrt(x@x); A1=2/r**2; C=-2*(r-2)/r**4; C1=(6*r-16)/r**5; px=p@x
    return (0.5*A1*(p@p)+0.5*C1*px*px)*x/r + C*px*p + x/r**3
def H(x,p):
    r=np.sqrt(x@x); A=(r-2)/r; C=-2*(r-2)/r**4
    return 0.5*A*(p@p)+0.5*C*(p@x)**2-1/r
x=np.array([7.,3.,1.]); v=np.array([0.05,0.3,-0.02]); p=p_from_v(x,v)
print('v roundtrip',dHdp(x,p)-v)
def glf(x,p,h,it=6):
    ph=p.copy()
    for _ in range(it): ph=p-0.5*h*dHdq(x,ph)
    x1=x.copy()
    for _ in range(it): x1=x+0.5*h*(dHdp(x,ph)+dHdp(x1,ph))
    p1=ph-0.5*h*dHdq(x1,ph)
    return x1,p1
def Lz(x,p): return np.linalg.norm(np.cross(x,p))
for eta in (0.02,0.01):
    r=10.0; vt=(r-2)/(r*np.sqrt(r-3)); x=np.array([r,0,0.]); p=p_from_v(x,np.array([0,vt,0.]))
    e0,l0=H(x,p),Lz(x,p); P=2*np.pi*r*r*np.sqrt(r-3)/(r-2); dt=eta*r**1.5; n=int(200*P/dt)
    for i in range(n): x,p=glf(x,p,dt)
    print('GLF circ eta',eta,'dE/E',(H(x,p)-e0)/abs(e0),'dL/L',(Lz(x,p)-l0)/l0)
rp,ra=25.,5000.; L=7.3537443433616145; f=ra/(ra-2); vt=L/(f*ra)
for eta in (0.02,0.005):
    x=np.array([ra,0,0.]); p=p_from_v(x,np.array([0,vt,0.])); e0,l0=H(x,p),Lz(x,p); t=0; n=0
    while t<792447.25*3:
        r=np.sqrt(x@x); dt=2.**12
        while dt>eta*r**1.5: dt/=2
        x,p=glf(x,p,dt); t+=dt; n+=1
    print('GLF eccentric eta',eta,'steps',n,'dE/E',(H(x,p)-e0)/abs(e0),'dL/L',(Lz(x,p)-l0)/l0, 'E0',e0)
