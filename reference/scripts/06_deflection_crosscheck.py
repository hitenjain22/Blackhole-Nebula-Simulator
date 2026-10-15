import numpy as np
from scipy.integrate import solve_ivp
def defl(b):
    def rhs(p,s): return [s[1], -s[0]+3*s[0]**2]
    def ev(p,s): return s[0]
    ev.terminal=True; ev.direction=-1
    sol=solve_ivp(rhs,[1e-9,20],[1e-9/b,1/b],events=ev,rtol=1e-13,atol=1e-15,method='DOP853')
    return sol.t_events[0][0]-np.pi
for b in (5.3,6.0,10.0,50.0,1000.0):
    d=defl(b); print(b,d,np.degrees(d))
