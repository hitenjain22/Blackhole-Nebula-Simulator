"""Render the frames of a planned shot (phases/phase-08-gas-rendering.md 8.1 pipeline, 8.5):
frame time -> gas state between snapshots (interpolate.py) -> textures (snapshot_scene.py) ->
GPU march along bent rays with the stars behind (gpu_march.py) -> display (gas_frames.py).

One exposure for the whole shot (no pumping): the median of the automatic exposures of a few
probe frames spread over the shot, times 2^stops from the shot's exposure keyframes.
"""

from pathlib import Path

import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.director import FramePlan, Shot, Timeline
from nebula.render.gas_frames import Exposure, auto_gas_exposure, auto_star_exposure, compose
from nebula.render.gpu_march import OUT_PICTURE, render_gas, upload_scene
from nebula.render.interpolate import load_snap, state_at
from nebula.render.march import GreyBlackbody
from nebula.render.snapshot_scene import run_eos_and_mass, textures_from_particles
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map

GAS_PERCENTILE, GAS_LEVEL = 99.9, 1.0  # as scripts/phase8_step4_first_frame.py
N_PROBES = 5


def camera_in_gas(plan: FramePlan, timeline: Timeline, margin: float = 1.5) -> np.ndarray:
    """Frames whose camera sits inside gas: within `margin` x h of a particle of the snapshot
    nearest in time. A ray then starts inside the gas, and nearby gas fills the frame as a
    blur (or blocks it), so shots should avoid it (director_preview.py reports these)."""
    return camera_gas_clearance(plan, timeline, margin)[1]


def camera_gas_clearance(plan: FramePlan, timeline: Timeline, margin: float = 1.5):
    """(distance from the camera to the nearest particle, inside-gas flag) per frame, from the
    snapshot nearest in time. Gas a few units from the camera fills the frame with large, fast,
    blurry blobs where the global grid's coarse cells show (first orbit_pass draft)."""
    dist = np.zeros(len(plan.sim_t))
    inside = np.zeros(len(plan.sim_t), bool)
    k_near = np.abs(timeline.times[None, :] - plan.sim_t[:, None]).argmin(axis=1)
    for k in np.unique(k_near):
        d = np.load(timeline.paths[k])
        x, h = d["x"].astype(float), d["h"].astype(float)
        for i in np.nonzero(k_near == k)[0]:
            r = np.linalg.norm(x - plan.position[i], axis=1)
            dist[i] = r.min()
            inside[i] = bool(np.any(r < margin * h))
    return dist, inside


class ShotRenderer:
    def __init__(self, shot: Shot, plan: FramePlan, timeline: Timeline, width=None, height=None,
                 n_sub=None, sky=None):  # fmt: skip
        self.shot, self.plan, self.timeline = shot, plan, timeline
        self.width, self.height = width or shot.width, height or shot.height
        self.n_sub = n_sub or shot.n_sub
        self.model = GreyBlackbody(kappa=shot.kappa)
        self.sky = sky if sky is not None else upload_cube_map(build_cube_map(StarFieldConfig()))
        self._snaps = {}
        self._runs = {}
        self.base = None

    def _snap(self, k):
        if k not in self._snaps:
            if len(self._snaps) >= 4:
                self._snaps.pop(min(self._snaps))  # frames move forward in time
            self._snaps[k] = load_snap(self.timeline.paths[k])
        return self._snaps[k]

    def _eos_mass(self, k):
        run_dir = Path(self.timeline.paths[k]).parent.parent
        if run_dir not in self._runs:
            self._runs[run_dir] = run_eos_and_mass(run_dir)
        return self._runs[run_dir]

    def picture(self, i: int) -> np.ndarray:
        """Linear GPU picture (H, W, 4) of frame i: (gas I, T x star RGB)."""
        t = float(self.plan.sim_t[i])
        times = self.timeline.times
        k = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
        a, b = self._snap(k), self._snap(k + 1)
        r_drop = 4.0  # accretion radius of every run so far (manifest config r_acc)
        ids, x, s = state_at(a, b, min(max(t, a.t), b.t), r_drop=r_drop)
        eos, mass = self._eos_mass(k)
        temp = snapshot_temperature(eos, s["rho"], s["u"])
        texs = textures_from_particles(x, s["h"], mass, temp, np.zeros_like(x))
        cam = self.plan.camera(i, self.width, self.height)
        pic, _ = render_gas(cam, upload_scene(texs, self.model), n_sub=self.n_sub, sky=self.sky,
                            out_mode=OUT_PICTURE)  # fmt: skip
        if not np.isfinite(pic).all():
            raise FloatingPointError(f"frame {i}: {(~np.isfinite(pic)).sum()} non-finite values")
        return pic

    def calibrate(self, probes=None):
        """Base exposure: median of the auto exposures of N_PROBES frames over the shot."""
        n = len(self.plan.video_t)
        idx = np.linspace(0, n - 1, N_PROBES).astype(int) if probes is None else probes
        pics = [self.picture(int(i)) for i in idx]
        gas = [auto_gas_exposure(p[..., 0], GAS_PERCENTILE, GAS_LEVEL) for p in pics]
        stars = [auto_star_exposure(p[..., 1:4]) for p in pics]
        self.base = Exposure(float(np.median(gas)), float(np.median(stars)))
        return self.base

    def frame(self, i: int) -> np.ndarray:
        """uint8 sRGB frame i."""
        if self.base is None:
            self.calibrate()
        # Stops change the gas only: the stars stay a steady backdrop (opening up 2 stops for
        # the cooling stream made the star field noisy and over-bright; orbit_pass draft 3).
        exp = Exposure(self.base.gas * 2.0 ** float(self.plan.stops[i]), self.base.stars)
        return compose(self.picture(i), exp)
