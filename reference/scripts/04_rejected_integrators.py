import numpy as np, time
def acc(x,v):
    r=np.sqrt(x@x); rd=x@v/r; c=np.cross(x,v); h2=c@c
    return -(1/r**3)*(1-2/r)**2*x + 2*rd/(r*(r-2))*v - 3*h2/r**5*x
def inv(x,v):
    r=np.sqrt(x@x); f=r/(r-2); rd=x@v/r
    eps=0.5*f*f*rd*rd+0.5*f*(v@v-rd*rd)-1/r
    ell=f*np.linalg.norm(np.cross(x,v)); return eps,ell
def step_implicit(x,v,dt,it=4):
    # v_half = v + dt/2 a(x, v_half)  (implicit), drift, v1 = vh + dt/2 a(x1, v1) implicit
    vh=v.copy()
    for _ in range(it): vh=v+0.5*dt*acc(x,vh)
    x1=x+dt*vh
    v1=vh.copy()
    for _ in range(it): v1=vh+0.5*dt*acc(x1,v1)
    return x1,v1
def step_explicit(x,v,dt):
    a=acc(x,v); vh=v+0.5*dt*a; x1=x+dt*vh
    vp=vh+0.5*dt*acc(x1,vh)   # predictor
    v1=vh+0.5*dt*acc(x1,vp)
    return x1,v1
for name,stepper in (('implicit',step_implicit),('explicit',step_explicit)):
  for eta in (0.02,0.01):
    r=10.0; vt=(r-2)/(r*np.sqrt(r-3))
    x=np.array([r,0,0.]); v=np.array([0,vt,0.]); e0,l0=inv(x,v)
    P=2*np.pi*r*r*np.sqrt(r-3)/(r-2); dt=eta*r**1.5; n=int(200*P/dt)
    for i in range(n): x,v=stepper(x,v,dt)
    e,l=inv(x,v); print(name,'circ r=10 eta',eta,'200 orbits: dE/E',(e-e0)/abs(e0),'dL/L',(l-l0)/l0,'r',np.sqrt(x@x))
# eccentric orbit, adaptive power-of-two dt chosen from r at step start
rp,ra=25.,5000.; L=7.3537443433616145; f=ra/(ra-2); vt=L/(f*ra)
for name,stepper in (('implicit',step_implicit),('explicit',step_explicit)):
  for eta in (0.02,0.005):
    x=np.array([ra,0,0.]); v=np.array([0,vt,0.]); e0,l0=inv(x,v); t=0; dtmax=2**12; peri=[]
    prev=0; rprev=ra; n=0
    while t<792447.25*3:
        r=np.sqrt(x@x); dt=dtmax
        while dt>eta*r**1.5: dt/=2
        x,v=stepper(x,v,dt); t+=dt; n+=1
    e,l=inv(x,v); print(name,'eccentric eta',eta,'3 radial periods steps',n,'dE/E',(e-e0)/abs(e0),'dL/L',(l-l0)/l0)
