import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import ive
print('== Sod exact (Toro), gamma 1.4')
g=1.4; rl,ul,pl=1.0,0.0,1.0; rr,ur,pr=0.125,0.0,0.1
cl=np.sqrt(g*pl/rl); cr=np.sqrt(g*pr/rr)
def fK(p,rk,pk,ck):
    if p>pk:
        A=2/((g+1)*rk); B=(g-1)/(g+1)*pk; return (p-pk)*np.sqrt(A/(p+B))
    return 2*ck/(g-1)*((p/pk)**((g-1)/(2*g))-1)
ps=brentq(lambda p: fK(p,rl,pl,cl)+fK(p,rr,pr,cr)+(ur-ul),1e-6,10,xtol=1e-15)
us=0.5*(ul+ur)+0.5*(fK(ps,rr,pr,cr)-fK(ps,rl,pl,cl))
rsl=rl*(ps/pl)**(1/g)
rsr=rr*((ps/pr+(g-1)/(g+1))/((g-1)/(g+1)*ps/pr+1))
S=ur+cr*np.sqrt((g+1)/(2*g)*ps/pr+(g-1)/(2*g))
csl=cl*(ps/pl)**((g-1)/(2*g))
print('p* %.6f u* %.6f rho*L %.6f rho*R %.6f shock speed %.6f head %.6f tail %.6f'%(ps,us,rsl,rsr,S,-cl,us-csl))
print('at t=0.2 (diaphragm x0=0): head %.5f tail %.5f contact %.5f shock %.5f'%(-cl*0.2,(us-csl)*0.2,us*0.2,S*0.2))
print('u (specific internal energy) left-of-contact %.6f right %.6f'%(ps/((g-1)*rsl),ps/((g-1)*rsr)))
print('== kernel cubic spline: W(0)=1/pi=%.10f, W(h)=0.25/pi=%.10f ; neighbours (4pi/3)(2*1.2)^3=%.4f'%(1/np.pi,0.25/np.pi,4*np.pi/3*(2*1.2)**3))
def W(q): return np.where(q<1,1-1.5*q*q+0.75*q**3,np.where(q<2,0.25*(2-q)**3,0))/np.pi
I,_=quad(lambda q:4*np.pi*q*q*W(q),0,2,points=[1]); print('int W dV = %.12f'%I)
print('== LBP ring (nu const), Sigma*pi*R0^2/m = (1/tau) x^-1/4 exp(-(1-x)^2/tau) ive(1/4,2x/tau)')
def lbp(x,tau): return (1/tau)*x**-0.25*np.exp(-(1-x)**2/tau)*ive(0.25,2*x/tau)/np.pi
for tau in (0.016,0.032,0.064,0.128,0.256,0.512,1.0,2.0):
    Mt,_=quad(lambda x:2*np.pi*x*lbp(x,tau),0,60,limit=1000,points=[1],epsabs=1e-13)
    print('tau %.3f mass %.9f  Sigma(x=0.5) %.6e Sigma(x=1) %.6e Sigma(x=1.5) %.6e'%(tau,Mt,lbp(0.5,tau),lbp(1.0,tau),lbp(1.5,tau)))
def mass(tau): return quad(lambda x:2*np.pi*x*lbp(x,tau),0,200,limit=2000,points=[1],epsabs=1e-13)[0]
th=brentq(lambda t: mass(t)-0.5,0.3,50); print('half-mass tau (nu const, inner edge at 0) %.5f'%th)
# angular momentum of LBP ring: J = int 2 pi x Sigma x^{1/2} dx conserved? (should be conserved = 1 for nu const LBP)
for tau in (0.016,0.256,1.0):
    J,_=quad(lambda x:2*np.pi*x*lbp(x,tau)*np.sqrt(x),0,200,limit=2000,points=[1],epsabs=1e-13)
    print('tau %.3f  angular momentum %.9f'%(tau,J))
