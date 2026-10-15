import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recon; recon.install(); recon.STORE['mode'] = int(sys.argv[1])
from nebula.sim.blob import hot_blob
f = hot_blob(0.00125, 0.15, block_steps=True); c = hot_blob(0.0025, 0.3, block_steps=True)
print(f"mode {sys.argv[1]} block blob: dE fine {f['max_energy_error']:.2e} coarse {c['max_energy_error']:.2e}; dL fine {f['max_angmom_error']:.1e} coarse {c['max_angmom_error']:.1e} ratio {c['max_angmom_error']/f['max_angmom_error']:.2f}; u_max/U0 {f['u_max']/4e-8:.0f}", flush=True)
