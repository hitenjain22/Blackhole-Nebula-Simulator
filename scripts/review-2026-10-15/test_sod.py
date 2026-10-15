"""Sod shock tube (quick) with and without the reconstruction prototype. Usage: test_sod.py recon|plain"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
mode = sys.argv[1]
if mode == 'recon':
    import recon; recon.install()
from nebula.sim import testproblems as tp
w = time.perf_counter()
prob = tp.sod_setup(sys.argv[2] if len(sys.argv) > 2 else 'quick')
st = tp.run_problem(prob)
m = tp.sod_measure(prob, st)
sol = tp.solve_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, tp.SOD_GAMMA)
np.savez(sys.argv[1] + '_' + (sys.argv[2] if len(sys.argv) > 2 else 'quick') + '.npz', x=st.x, v=st.v, rho=st.rho, u=st.u, t=st.t)
print(mode, f"wall {time.perf_counter()-w:.0f}s")
print(f"  rho*L {m.rho_star_l:.4f} (exact {sol.rho_star_l:.4f})  rho*R {m.rho_star_r:.4f} (exact {sol.rho_star_r:.4f})  p* {m.p_star:.4f} (exact {sol.p_star:.4f})  u* {m.u_star:.4f} (exact {sol.u_star:.4f})")
print(f"  shock x {m.shock_x:.4f} (exact {m.shock_x_exact:.4f}, h {m.h_shock:.4f})  post-shock oscillation {m.oscillation:.3f}")
x = st.x[:, 0]; edges = np.linspace(0, prob.extra['length'], 401); c, rb = tp._bin(x, st.rho, edges)
ex = tp.sod_exact(prob, c, st.t)
rho_ex = ex[0] if isinstance(ex, tuple) else ex['rho'] if isinstance(ex, dict) else ex
try:
    print(f"  L1(rho) over the tube: {np.nanmean(np.abs(rb - rho_ex)):.4f}")
except Exception as e:
    print('  L1 skipped', e)
