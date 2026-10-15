import numpy as np
from scipy.special import iv
from scipy.integrate import quad
# Doppler circular orbit r=10, tangential photon emission L_z=±bmax
r=10; ut=1/np.sqrt(1-3/r); Om=r**-1.5; bmax=r/np.sqrt(1-2/r)
for b in (bmax,-bmax,0):
    print('g(b=%.4f)'%b, 1/(ut*(1-Om*b)))
# LBP ring normalization
def sigma(x,tau,m=1,R0=1): return m/(np.pi*R0**2)/tau*x**-0.25*np.exp(-(1+x*x)/tau)*iv(0.25,2*x/tau)
for tau in (0.016,0.064,0.256):
    M,_=quad(lambda x:2*np.pi*x*sigma(x,tau),1e-8,20,limit=500,points=[1])
    print('tau',tau,'mass',M)
# Planckian chromaticity via Wyman 2013 multi-lobe fit
def g(x,mu,s1,s2): return np.exp(-0.5*((x-mu)/np.where(x<mu,s1,s2))**2)
def cmf(l):
    xb=1.056*g(l,599.8,37.9,31.0)+0.362*g(l,442.0,16.0,26.7)-0.065*g(l,501.1,20.4,26.2)
    yb=0.821*g(l,568.8,46.9,40.5)+0.286*g(l,530.9,16.3,31.1)
    zb=1.217*g(l,437.0,11.8,36.0)+0.681*g(l,459.0,26.0,13.8)
    return xb,yb,zb
l=np.linspace(360,830,4701)
h=6.62607015e-34;c=2.99792458e8;k=1.380649e-23
def xy(T):
    lam=l*1e-9; B=1/lam**5/np.expm1(h*c/(lam*k*T))
    X,Y,Z=[np.trapezoid(B*f,l) for f in cmf(l)]
    return X/(X+Y+Z),Y/(X+Y+Z)
for T in (1000,2000,3000,6500,10000,30000,1e5,1e7):
    print(T, xy(T))
