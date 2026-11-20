"""Extra rays on the photon ring (DECISIONS 2026-11-19).

The lensed image of the inner gas squeezed against the shadow edge (the photon ring) is thinner
than a pixel. With 4 rays per pixel some pixels caught it and some missed: a dotted, even,
1-px circle that looked drawn on (outputs/checks/photon_ring_zoom_dotted.png). Pixels whose
centre ray passes within ``width_px`` pixels of the critical impact parameter b_crit = 3 sqrt(3)
(physics.geodesics.B_CRIT; b from physics.geodesics.schw_impact_parameter, float64) get
``n_ring`` x ``n_ring`` rays instead; the map is computed here and read by the GPU shader
(gas_march.wgsl binding 16), and the CPU reference (bent_rays.render_reference) uses it too.
"""

import numpy as np

from nebula.physics.geodesics import B_CRIT, schw_impact_parameter
from nebula.render.camera import Camera

N_RING = 8  # rays per side on ring pixels (64 rays), the shader's MAX_RING_SUB
WIDTH_PX = 2.0  # ring band half-width in pixels of impact parameter


def impact_parameter_image(camera: Camera, px, py) -> np.ndarray:
    """Impact parameter b (GM/c^2) of the rays through continuous pixel coordinates."""
    d = camera.directions(px, py)
    pos = np.asarray(camera.position, float)
    r_cam = float(np.linalg.norm(pos))
    psi = np.arccos(np.clip(d @ (-pos / r_cam), -1.0, 1.0))
    return schw_impact_parameter(r_cam, psi)


def ring_subsample_map(camera: Camera, n_ring: int = N_RING, width_px: float = WIDTH_PX):
    """(H, W) uint32: n_ring on pixels whose centre b is within width_px pixels of b_crit
    (pixel scale of b from the neighbouring pixel centres), 0 elsewhere."""
    w, h = camera.width, camera.height
    px, py = np.meshgrid(np.arange(-1, w + 1) + 0.5, np.arange(-1, h + 1) + 0.5)
    b = impact_parameter_image(camera, px, py)
    c = b[1:-1, 1:-1]
    db = np.maximum.reduce([np.abs(b[1:-1, 2:] - c), np.abs(b[1:-1, :-2] - c),
                            np.abs(b[2:, 1:-1] - c), np.abs(b[:-2, 1:-1] - c)])  # fmt: skip
    on = np.abs(c - B_CRIT) <= width_px * db
    return np.where(on, np.uint32(n_ring), np.uint32(0))
