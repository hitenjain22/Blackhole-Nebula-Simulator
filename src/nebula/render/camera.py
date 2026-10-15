"""Pinhole camera (phases/phase-02-blackhole-render.md Step 2.3).

The camera is a **static observer** (it hovers at a fixed point; a moving camera would add
aberration, which is added later only if a shot needs it, with a test). Viewing directions
are unit vectors in the static observer's own orthonormal frame, whose axes are aligned
with the coordinate x, y, z directions; the ray tracer turns them into impact parameters
(physics/geodesics.py). The WGSL copy of ``directions`` is ``pixel_dir`` in
shaders/camera_rays.wgsl (tested against it).

Image convention: pixel (i, j) has column i (0 = left) and row j (0 = top); continuous pixel
coordinates run from 0 to width / height, so pixel centres are at i + 0.5, j + 0.5.
"""

from dataclasses import dataclass

import numpy as np

# Render modes understood by shaders/camera_rays.wgsl and its entry shaders.
MODE_FLAT_GNOMONIC_GRID = 0  # straight rays, grid of straight lines on the sky (camera test)
MODE_HOLE_LATLONG_GRID = 1  # bent rays, latitude-longitude grid on the sky
MODE_FLAT_LATLONG_GRID = 2  # straight rays, latitude-longitude grid
MODE_HOLE_STARS = 3  # bent rays, star-field cube map (Step 2.4)
MODE_FLAT_STARS = 4  # straight rays, star-field cube map
MODE_DEBUG_DIRECTION = 10  # output the world direction of the pixel centre
MODE_DEBUG_GNOMONIC = 11  # output (d_x/d_z, d_y/d_z, d_z) of the pixel centre, straight rays


@dataclass(frozen=True)
class Camera:
    """A static pinhole camera at ``position`` looking at ``look_at``.

    ``up`` fixes the roll (it need not be perpendicular to the view direction); ``vfov_deg``
    is the vertical field of view. Width and height must be even (video encoding).
    """

    position: tuple[float, float, float]
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vfov_deg: float = 60.0
    width: int = 1280
    height: int = 720

    def __post_init__(self):
        if np.linalg.norm(self.position) <= 3.0:
            raise ValueError("camera must be outside the photon sphere (r > 3)")
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even")
        if abs(np.dot(self.forward, np.asarray(self.up, float) / np.linalg.norm(self.up))) > 0.999:
            raise ValueError("up vector is (nearly) parallel to the view direction")

    @property
    def forward(self) -> np.ndarray:
        f = np.asarray(self.look_at, float) - np.asarray(self.position, float)
        return f / np.linalg.norm(f)

    @property
    def right(self) -> np.ndarray:
        r = np.cross(self.forward, np.asarray(self.up, float))
        return r / np.linalg.norm(r)

    @property
    def up_cam(self) -> np.ndarray:
        """The camera's own up axis: perpendicular to forward and right."""
        return np.cross(self.right, self.forward)

    @property
    def tan_half_vfov(self) -> float:
        return float(np.tan(np.radians(self.vfov_deg) / 2))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def directions(self, px, py) -> np.ndarray:
        """Unit viewing directions (..., 3) for continuous pixel coordinates px, py."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        x = (2.0 * px / self.width - 1.0) * self.tan_half_vfov * self.aspect
        y = (1.0 - 2.0 * py / self.height) * self.tan_half_vfov
        d = self.forward + x[..., None] * self.right + y[..., None] * self.up_cam
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def pixel_of_direction(self, d) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of ``directions`` (continuous pixel coordinates); for directions in front."""
        d = np.asarray(d, float)
        z = d @ self.forward
        x = (d @ self.right) / z / (self.tan_half_vfov * self.aspect)
        y = (d @ self.up_cam) / z / self.tan_half_vfov
        return (x + 1.0) * self.width / 2.0, (1.0 - y) * self.height / 2.0

    def uniform_bytes(self, n_sub: int, mode: int, sky_params=(0.0, 0.0)) -> bytes:
        """The ``Camera`` uniform of shaders/camera_rays.wgsl: n_sub x n_sub sub-rays per
        pixel on a fixed stratified pattern; ``sky_params`` are mode-specific."""
        f32 = np.zeros(20, dtype=np.float32)
        f32[0:3] = self.position
        f32[4:7] = self.right
        f32[8:11] = self.up_cam
        f32[12:15] = self.forward
        f32[16:20] = (self.tan_half_vfov, self.aspect, *sky_params)
        u32 = np.array([self.width, self.height, n_sub, mode], dtype=np.uint32)
        return f32.tobytes() + u32.tobytes()


def sub_pixel_offsets(n_sub: int) -> np.ndarray:
    """The fixed n_sub x n_sub stratified sub-ray pattern, offsets in [0, 1)^2 (same every
    frame, so nothing flickers). Mirrors the sub-ray loops of the entry shaders."""
    k = (np.arange(n_sub) + 0.5) / n_sub
    a, b = np.meshgrid(k, k, indexing="xy")
    return np.column_stack([a.ravel(), b.ravel()])
