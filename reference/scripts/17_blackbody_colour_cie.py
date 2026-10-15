import warnings; warnings.filterwarnings('ignore')
import numpy as np, colour
cm=colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
wl=cm.wavelengths; xb,yb,zb=cm.values.T
sel=(wl>=360)&(wl<=830); wl,xb,yb,zb=wl[sel],xb[sel],yb[sel],zb[sel]
print('CMF table: %d samples %g..%g nm step %g'%(len(wl),wl[0],wl[-1],wl[1]-wl[0]))
h=6.62607015e-34;c=2.99792458e8;k=1.380649e-23;sigma=5.670374419e-8
M=colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
print('XYZ->linear sRGB matrix:\n',np.array2string(M,precision=7))
def XYZ(T):
    lam=wl*1e-9
    if np.isinf(T): B=lam**-4.0   # Rayleigh-Jeans limit
    else: B=2*h*c**2/lam**5/np.expm1(h*c/(lam*k*T))
    return np.array([np.sum(B*xb),np.sum(B*yb),np.sum(B*zb)])*1e-9  # 1 nm steps
print('T K | x y (CIE1931 2deg, 1nm, 360-830) | linear sRGB normalised to max=1 | luminous efficacy lm/W')
for T in (1000,1500,2000,3000,4000,5000,5778,6500,10000,20000,30000,1e5,1e6,np.inf):
    X=XYZ(T); s=X.sum(); x,y=X[0]/s,X[1]/s
    rgb=M@X; rgbn=rgb/rgb.max()
    eff='' if np.isinf(T) else '%.2f'%(683.002*X[1]/(sigma*T**4/np.pi))
    print('%8s | %.5f %.5f | %s | %s'%(T,x,y,np.array2string(rgbn,precision=4),eff))
# compare with colour-science's own planckian chromaticity
try:
    from colour.temperature import CCT_to_xy_Planck
except Exception as e: CCT_to_xy_Planck=None
for T in (2000,6500):
    sd=colour.sd_blackbody(T, colour.SpectralShape(360,830,1))
    X=colour.sd_to_XYZ(sd,cm,method='Integration'); print('colour-science sd_to_XYZ T=%d xy='%T,np.round(X[:2]/X.sum(),5))
