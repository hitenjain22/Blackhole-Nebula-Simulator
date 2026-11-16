"""Detail boxes far from the hole (phases/phase-08-gas-rendering.md 8.3 item 2; Phase 8 step 6b)."""

import numpy as np

from nebula.render.bent_rays import march_bent_ray
from nebula.render.gasgrid import LogSphericalGrid
from nebula.render.march import GreyBlackbody
from nebula.render.snapshot_scene import inside_boxes, textures_from_particles
from tests.helpers import check

CLOUD = np.array([5000.0, 0.0, 0.0])


def _cloud():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(3000, 3))
    x = CLOUD + 60.0 * v / np.linalg.norm(v, axis=1)[:, None] * rng.random((3000, 1)) ** (1 / 3)
    h = np.full(3000, 3.0)
    return x, h, np.full(3000, 1.0 / 3000), np.full(3000, 1e4), np.zeros((3000, 3))


def test_cloud_inside_a_far_box_leaves_no_fog():
    """A cloud of radius 60 at r = 5,000 inside a 400-wide box: every particle is deep inside,
    so the log grid (cells ~200 wide out there) holds none of its mass. A ray through the
    cloud sees it; a ray passing 100 beyond the box face sees nothing (before the fix the log
    grid's smeared copy of the cloud glowed around the box and the camera)."""
    x, h, m, t, v = _cloud()
    boxes = [{"centre": tuple(CLOUD), "size": 400.0, "n": 96}]
    check("far box: all cloud particles deep inside", 3000, int(inside_boxes(x, h, boxes).sum()))
    texs = textures_from_particles(x, h, m, t, v, LogSphericalGrid(r_max=8000.0, nr=230), boxes)
    check("far box: no cloud mass in the log grid", 0.0, float(texs[0].rho.max()))
    model = GreyBlackbody(kappa=1e4)
    cam = CLOUD + [0.0, 300.0, 0.0]
    i_hit, t_hit, _, _ = march_bent_ray(texs, model, cam, np.array([0.0, -1.0, 0.0]))
    miss = CLOUD + [0.0, 0.0, 300.0] - cam
    i_miss, t_miss, _, _ = march_bent_ray(texs, model, cam, miss / np.linalg.norm(miss))
    check("far box: ray through the cloud sees gas (transmission < 1)", True,
          bool(i_hit > 0 and t_hit < 0.999))  # fmt: skip
    check("far box: ray beside the box sees nothing", (0.0, 1.0), (i_miss, t_miss))


def test_particles_near_a_box_face_stay_in_the_log_grid():
    """Particles whose kernel reaches the box's blending zone are kept in the log grid too
    (there the two are blended, so both must hold them)."""
    x = np.array([[5000.0, 0, 0], [5000.0 + 190.0, 0, 0], [5000.0 + 150.0, 0, 0]])
    boxes = [{"centre": tuple(CLOUD), "size": 400.0, "n": 100}]  # dx 4: reach 2h + 3 dx
    deep = inside_boxes(x, np.array([3.0, 3.0, 3.0]), boxes)
    check("near face: centre deep, 10 from the face kept, 50 from the face deep",
          (True, False, True), tuple(bool(d) for d in deep))  # fmt: skip
