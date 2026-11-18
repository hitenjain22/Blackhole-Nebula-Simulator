"""Render the frames of a planned shot (phases/phase-08-gas-rendering.md 8.1 pipeline, 8.5):
frame time -> gas state between snapshots (interpolate.py) -> textures (snapshot_scene.py) ->
GPU march along bent rays with the stars behind (gpu_march.py) -> display (gas_frames.py).

One exposure for the whole shot (no pumping): the median of the automatic exposures of a few
probe frames spread over the shot, times 2^stops from the shot's exposure keyframes.
"""

from pathlib import Path

import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.physics.colour import chroma_lut, srgb_decode, srgb_encode
from nebula.physics.redshift import gn_static_frame_velocity
from nebula.render.colour_data import load_cmf
from nebula.render.director import CODE_DAY, FramePlan, Shot, Timeline, fade_factor
from nebula.render.frames import to_uint8
from nebula.render.gas_frames import (
    Exposure,
    auto_gas_exposure,
    auto_star_exposure,
    compose,
    compose_colour,
)
from nebula.render.gpu_march import OUT_COLOUR_PICTURE, OUT_PICTURE, render_gas, upload_scene
from nebula.render.interpolate import load_snap
from nebula.render.junctions import JunctionEaser, eased_state
from nebula.render.march import GreyBlackbody, RelativisticBlackbody
from nebula.render.snapshot_scene import inside_boxes, run_eos_and_mass, textures_from_particles
from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map

GAS_PERCENTILE, GAS_LEVEL = 99.9, 1.0  # as scripts/phase8_step4_first_frame.py
N_PROBES = 5


def adaptation_stops(probe_t, probe_gas_exposure, frame_t, smooth_s: float, strength: float):
    """Exposure change per frame, in stops, from automatic exposures measured at probe times:
    strength x log2(probe exposure / median probe exposure), smoothed by a Gaussian of width
    smooth_s seconds of video time (an eye adapting: no sudden changes) and read at the frame
    times. strength < 1 adapts only partly, so very hot gas still looks brighter than cool."""
    pt = np.asarray(probe_t, float)
    stops = strength * np.log2(np.asarray(probe_gas_exposure, float))
    stops -= np.median(stops)
    # Odd reflection at both ends (t -> 2 t_end - t, s -> 2 s_end - s): a plain Gaussian mean
    # near an end only sees one side and flattens a trend there (first low-res final render:
    # half the needed correction in the first 3 s); with the reflection a steady trend is
    # followed exactly up to the ends.
    # reflected over 6 widths: a steady trend is then followed to ~1e-8 of a stop, and steps
    # farther away than that are not mirrored back in (no overshoot)
    near0 = pt[1:] <= pt[0] + 6.0 * smooth_s
    near1 = pt[:-1] >= pt[-1] - 6.0 * smooth_s
    t_all = np.concatenate([2 * pt[0] - pt[1:][near0], pt, 2 * pt[-1] - pt[:-1][near1]])
    s_all = np.concatenate([2 * stops[0] - stops[1:][near0], stops,
                            2 * stops[-1] - stops[:-1][near1]])  # fmt: skip
    ft = np.asarray(frame_t, float)
    w = np.exp(-0.5 * ((ft[:, None] - t_all[None, :]) / smooth_s) ** 2)
    return (w @ s_all) / w.sum(axis=1)


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


def gas_outside_boxes(plan: FramePlan, shot: Shot, timeline: Timeline) -> np.ndarray:
    """Per frame: the fraction of particles of the snapshot nearest in time that are NOT deep
    inside a detail box (so they go to the coarse log grid). A box that follows a far cloud
    must hold nearly all of it, or the outside part shows as a blurry fog (approach drafts).
    0 when the shot has no boxes is meaningless: returns ones then."""
    out = np.ones(len(plan.sim_t))
    if not shot.boxes:
        return out
    k_near = np.abs(timeline.times[None, :] - plan.sim_t[:, None]).argmin(axis=1)
    for k in np.unique(k_near):
        d = np.load(timeline.paths[k])
        x, h = d["x"].astype(float), d["h"].astype(float)
        for i in np.nonzero(k_near == k)[0]:
            out[i] = 1.0 - inside_boxes(x, h, plan.boxes(i, shot)).mean()
    return out


class ShotRenderer:
    def __init__(self, shot: Shot, plan: FramePlan, timeline: Timeline, width=None, height=None,
                 n_sub=None, sky=None):  # fmt: skip
        self.shot, self.plan, self.timeline = shot, plan, timeline
        self.width, self.height = width or shot.width, height or shot.height
        self.n_sub = n_sub or shot.n_sub
        self.colour = shot.colour == "true"
        if self.colour:  # Phase 9 true colour; r_cam is the GPU's own camera distance
            lam, cmf = load_cmf()
            lt, lc = chroma_lut(1e3, 1e8, 4096, lam, cmf)
            self.model = RelativisticBlackbody(kappa=shot.kappa, r_cam=float("inf"), log_t=lt,
                                               log_chroma=lc, opacity=shot.opacity)  # fmt: skip
        else:
            self.model = GreyBlackbody(kappa=shot.kappa)
        self.grid = shot.log_grid()  # same grid settings every frame (8.8: no flicker)
        self.sky = sky if sky is not None else upload_cube_map(build_cube_map(StarFieldConfig()))
        self._snaps = {}
        self._runs = {}
        self.easer = JunctionEaser(shot.sources, timeline, load_snap)  # joins between runs
        self.base = None
        self.adapt = np.zeros(len(plan.video_t))  # automatic exposure stops (adapt_exposure)

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
        """Linear GPU picture of frame i: grey (H, W, 4) = (gas I, T x star RGB); true colour
        (H, W, 8) = (X, Y, Z, bolometric, T x star RGB, 0)."""
        t = float(self.plan.sim_t[i])
        times = self.timeline.times
        k = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
        a, b = self._snap(k), self._snap(k + 1)
        r_drop = 4.0  # accretion radius of every run so far (manifest config r_acc)
        tc = min(max(t, a.t), b.t)
        ids, x, p, s = eased_state(a, b, tc, self.easer.at(tc), r_drop=r_drop, with_p=True)
        # the gas velocity (gamma beta) feeds the Doppler colours; grey ignores it
        v = gn_static_frame_velocity(x, p)[1] if self.colour else np.zeros_like(x)
        bad = ~np.isfinite(v).all(axis=1)
        if bad.any():  # no velocity inside r = 2; reported, not hidden (AI_LOG 2026-11-18)
            r_bad = np.linalg.norm(x[bad], axis=1)
            print(f"  frame {i}: dropped {bad.sum()} particle(s) with no velocity, "
                  f"r = {r_bad.min():.2f}", flush=True)  # fmt: skip
            x, v, s = x[~bad], v[~bad], {k: val[~bad] for k, val in s.items()}
        eos, mass = self._eos_mass(k)
        temp = snapshot_temperature(eos, s["rho"], s["u"])
        boxes = self.plan.boxes(i, self.shot)
        texs = textures_from_particles(x, s["h"], mass, temp, v, self.grid, boxes,
                                       self.shot.brick_lattice(t / CODE_DAY))  # fmt: skip
        cam = self.plan.camera(i, self.width, self.height)
        mode = OUT_COLOUR_PICTURE if self.colour else OUT_PICTURE
        pic, _ = render_gas(cam, upload_scene(texs, self.model), n_sub=self.n_sub, sky=self.sky,
                            out_mode=mode)  # fmt: skip
        if not np.isfinite(pic).all():
            raise FloatingPointError(f"frame {i}: {(~np.isfinite(pic)).sum()} non-finite values")
        return pic

    def calibrate(self, probes=None):
        """Base exposure: median of the auto exposures of N_PROBES frames over the shot."""
        n = len(self.plan.video_t)
        idx = np.linspace(0, n - 1, N_PROBES).astype(int) if probes is None else probes
        pics = [self.picture(int(i)) for i in idx]
        gas = [auto_gas_exposure(self._gas_level(p), GAS_PERCENTILE, GAS_LEVEL) for p in pics]
        stars = [auto_star_exposure(self._stars(p)) for p in pics]
        self.base = Exposure(float(np.median(gas)), float(np.median(stars)))
        return self.base

    def _gas_level(self, pic):
        """The gas brightness the exposure is set on: grey I, or luminance Y in true colour."""
        return pic[..., 1] if self.colour else pic[..., 0]

    def _stars(self, pic):
        return pic[..., 4:7] if self.colour else pic[..., 1:4]

    def adapt_exposure(self, every_s=1.0, smooth_s=4.0, strength=0.75, probe_size=(320, 180)):
        """Base exposure (median of the probes) plus a smooth automatic exposure curve
        (adaptation_stops) from low-resolution probe frames every ``every_s`` seconds; for long
        shots whose gas brightness changes by orders of magnitude (cool cloud -> hot pass)."""
        n = len(self.plan.video_t)
        idx = np.unique(np.round(np.arange(0.0, self.shot.duration, every_s) * self.shot.fps)
                        .astype(int).clip(0, n - 1))  # fmt: skip
        size, n_sub = (self.width, self.height), self.n_sub
        self.width, self.height = probe_size
        self.n_sub = 1
        try:
            pics = [self.picture(int(i)) for i in idx]
        finally:
            (self.width, self.height), self.n_sub = size, n_sub
        gas = np.array([auto_gas_exposure(self._gas_level(p), GAS_PERCENTILE, GAS_LEVEL)
                        for p in pics])  # fmt: skip
        stars = [auto_star_exposure(self._stars(p)) for p in pics]
        self.base = Exposure(float(np.median(gas)), float(np.median(stars)))
        self.adapt = adaptation_stops(self.plan.video_t[idx], gas, self.plan.video_t, smooth_s,
                                      strength)  # fmt: skip
        return self.base, self.adapt

    def frame(self, i: int) -> np.ndarray:
        """uint8 sRGB frame i."""
        if self.base is None:
            self.calibrate()
        # Stops change the gas only: the stars stay a steady backdrop (opening up 2 stops for
        # the cooling stream made the star field noisy and over-bright; orbit_pass draft 3).
        stops = float(self.plan.stops[i]) + float(self.adapt[i])
        exp = Exposure(self.base.gas * 2.0**stops, self.base.stars)
        img = (compose_colour if self.colour else compose)(self.picture(i), exp)
        fade = float(fade_factor(self.plan.video_t[i], self.plan.video_t[-1], self.shot.fade_out))
        if fade < 1.0:  # end fade, in linear light so it darkens evenly (stars too)
            img = to_uint8(srgb_encode(srgb_decode(img / 255.0) * fade))
        return img
