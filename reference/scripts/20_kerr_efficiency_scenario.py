import numpy as np
tu=21.180252
def isco(a):
    Z1=1+(1-a*a)**(1/3)*((1+a)**(1/3)+(1-a)**(1/3)); Z2=np.sqrt(3*a*a+Z1*Z1)
    return 3+Z2-np.sqrt((3-Z1)*(3+Z1+2*Z2))
for a in (0.,0.5,0.9,0.99):
    r=isco(a); E=(r**1.5-2*r**0.5+a)/(r**0.75*np.sqrt(r**1.5-3*r**0.5+2*a))
    print('a %.2f ISCO %.6f E_isco %.6f efficiency %.4f%%'%(a,r,E,100*(1-E)))
# torus nodal precession period at r_circ, a=0.9 (exact Kerr), in time units and days
a=0.9; r=50.8896237; Op=1/(r**1.5+a); On=Op*(1-np.sqrt(1-4*a*r**-1.5+3*a*a/r**2))
print('torus nodal precession at r_circ, a=0.9: rate %.4e, period %.4e = %.1f days'%(On,2*np.pi/On,2*np.pi/On*tu/86400))
# vertical miss distance at r_SI after one passage for tilt i
for inc in (30,60):
    print('tilt %d: offset at r_SI = r_SI * dOmega * sin(i) = %.1f (weak-field LT dOmega=0.03223)'%(inc,1663.756*0.03223*np.sin(np.radians(inc))))
# Eddington numbers
G=6.67430e-11;c=2.99792458e8;Msun=1.98847e30;kes=0.034
M=4.3e6*Msun; mp=1.67262192369e-27; sigT=6.6524587321e-29
LEdd_sol=4*np.pi*G*M*c/kes
LEdd=4*np.pi*G*M*mp*c/sigT   # standard pure-hydrogen convention used in the plan's reports
print('L_Edd (pure H, used for reports) %.4e W ; Mdot_Edd(eta=0.1) %.4e kg/s = %.5f Msun/yr'%(LEdd,LEdd/(0.1*c*c),LEdd/(0.1*c*c)*3.15576e7/Msun))
print('L_Edd with solar kappa_es=0.034 (NOT used for reports) %.4e W'%LEdd_sol)
E_diss=(-1.98918881503e-04 - (-9.620022495495e-03)); print('dissipated per unit mass %.5e c^2 ; for 0.1 Msun %.4e J ; in L_Edd-days %.1f'%(E_diss,E_diss*0.1*Msun*c*c,E_diss*0.1*Msun*c*c/LEdd/86400))
# cloud density, optical depth at start
lu=G*M/c**2; R=100*lu; rho=0.1*Msun/(4/3*np.pi*R**3); print('cloud rho %.4e kg/m3, tau_es(R) %.1f'%(rho,kes*rho*R))
# photosphere effective temperature estimate for a torus radiating L_Edd
Rt=50.89*lu; at=0.3*Rt; A=4*np.pi**2*Rt*at; print('torus surface area %.3e m2 ; T_eff at L_Edd %.3e K'%(A,(LEdd/(5.670374419e-8*A))**0.25))
