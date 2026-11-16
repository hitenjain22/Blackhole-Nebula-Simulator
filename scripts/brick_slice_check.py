"""Sparse brick grid check picture (DECISIONS 2026-11-16): density in the stream's orbital
plane from the log grid alone (as trailer_v1) and from log grid + bricks, side by side.

    uv run python scripts/brick_slice_check.py [run=dense_B192] [day=97.70] [dx=1.0] [half=450]

Writes outputs/checks/bricks_slice_<run>_day<day>.png (log10 density, same colour scale)."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.render.bricks import BRICK, BrickLattice, deposit_bricks, fine_particles, pack_keys
from nebula.render.director import CODE_DAY, Timeline
from nebula.render.gasgrid import LogSphericalGrid, deposit_log
from nebula.render.snapshot_scene import run_eos_and_mass

ROOT = Path(__file__).resolve().parents[1]
N_PIX = 900


def log_slice(gg, pts):
    """Vectorised trilinear of the log grid's rho (mirrors march.sample_log; picture only)."""
    g = gg.grid
    r = np.linalg.norm(pts, axis=1)
    out = np.zeros(len(pts))
    ok = (r >= g.r_min) & (r <= g.r_max)
    p, r = pts[ok], r[ok]
    fi = np.log(r / g.r_min) / g.dlnr - 0.5
    fj = np.arccos(np.clip(p[:, 2] / r, -1, 1)) / g.dth - 0.5
    fk = (np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)) / g.dph - 0.5
    i0, j0, k0 = (np.floor(a).astype(int) for a in (fi, fj, fk))
    acc = np.zeros(len(p))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((fi - i0 if di else 1 - (fi - i0)) * (fj - j0 if dj else 1 - (fj - j0))
                     * (fk - k0 if dk else 1 - (fk - k0)))  # fmt: skip
                ii = np.clip(i0 + di, 0, g.nr - 1)
                jj = np.clip(j0 + dj, 0, g.nth - 1)
                kk = (k0 + dk) % g.nph
                acc += w * gg.rho[ii, jj, kk]
    out[ok] = acc
    return out


def brick_slice(bg, pts):
    """Vectorised trilinear of the bricks' rho (mirrors bricks.sample_bricks)."""
    dx = bg.lattice.dx
    f = pts / dx - 0.5
    i0 = np.floor(f).astype(np.int64)
    t = f - i0
    out = np.zeros(len(pts))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                c = i0 + [di, dj, dk]
                w = np.prod(np.where([di, dj, dk], t, 1 - t), axis=1)
                key = pack_keys(c[:, 0] // BRICK, c[:, 1] // BRICK, c[:, 2] // BRICK)
                s = np.clip(np.searchsorted(bg.keys, key), 0, len(bg.keys) - 1)
                hit = bg.keys[s] == key
                v = np.zeros(len(pts))
                v[hit] = bg.rho[s[hit], c[hit, 0] % BRICK, c[hit, 1] % BRICK, c[hit, 2] % BRICK]
                out += w * v
    return out


def main():
    o = dict(a.split("=", 1) for a in sys.argv[1:])
    run, day = o.get("run", "dense_B192"), float(o.get("day", 97.70))
    lat = BrickLattice(dx=float(o.get("dx", 1.0)))
    half = float(o.get("half", 450))
    tl = Timeline.from_sources([{"run": run}], ROOT / "runs")
    k = int(np.argmin(np.abs(tl.times - day * CODE_DAY)))
    d = np.load(tl.paths[k])
    x, h = d["x"].astype(float), d["h"].astype(float)
    eos, mass = run_eos_and_mass(ROOT / "runs" / run)
    temp = snapshot_temperature(eos, d["rho"].astype(float), d["u"].astype(float))
    g = LogSphericalGrid()
    t0 = time.perf_counter()
    old = deposit_log(x, h, mass, temp, np.zeros_like(x), g)
    t1 = time.perf_counter()
    fine = fine_particles(x, h, lat, g)
    rest = deposit_log(x[~fine], h[~fine], mass, temp[~fine], np.zeros_like(x[~fine]), g)
    bg = deposit_bricks(x[fine], h[fine], mass, temp[fine], lat)
    t2 = time.perf_counter()
    print(
        f"day {float(d['t']) / CODE_DAY:.2f}: {fine.mean() * 100:.1f}% of particles on "
        f"{len(bg.keys)} bricks ({len(bg.keys) * 729 * 8 / 1e6:.0f} MB on the GPU); deposit "
        f"log only {t1 - t0:.1f} s, log + bricks {t2 - t1:.1f} s; mass check "
        f"{(rest.mass.sum() + rest.outside + bg.mass.sum()) / (len(h) * mass):.12f}"
    )
    # plane through the hole perpendicular to the gas's angular momentum (the orbital plane)
    vel = d["p"].astype(float) if "p" in d else np.zeros_like(x)
    ang = np.cross(x, vel).sum(axis=0)
    nz = ang / np.linalg.norm(ang)
    ex = np.cross(nz, [0.0, 0.0, 1.0] if abs(nz[2]) < 0.9 else [1.0, 0.0, 0.0])
    ex /= np.linalg.norm(ex)
    ey = np.cross(nz, ex)
    # centre the picture on the stream (mass centre of the fine gas, projected on the plane)
    c = x[fine].mean(axis=0)
    c = c - (c @ nz) * nz
    u = np.linspace(-half, half, N_PIX)
    uu, vv = np.meshgrid(u, u)
    pts = c + uu.reshape(-1, 1) * ex + vv.reshape(-1, 1) * ey
    a = log_slice(old, pts).reshape(N_PIX, N_PIX)
    b = (log_slice(rest, pts) + brick_slice(bg, pts)).reshape(N_PIX, N_PIX)
    vmax = np.log10(max(a.max(), b.max()))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8.4))
    titles = ["before: log grid only (trailer v1)", f"after: log grid + bricks (dx = {lat.dx:g})"]
    for axis, img, title in zip(ax, (a, b), titles, strict=True):
        axis.imshow(np.log10(np.maximum(img, 1e-30)), origin="lower", cmap="inferno",
                    vmin=vmax - 4, vmax=vmax, extent=[-half, half, -half, half])  # fmt: skip
        axis.set_title(title)
        axis.set_xlabel("GM/c^2 (in the orbital plane)")
    fig.suptitle(f"{run}, day {float(d['t']) / CODE_DAY:.2f}: gas density in the stream's "
                 "orbital plane (4 decades)")  # fmt: skip
    plt.tight_layout()
    out = ROOT / "outputs" / "checks" / f"bricks_slice_{run}_day{day:.2f}.png"
    plt.savefig(out, dpi=80)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
