"""A cinematic preview of a run: glowing gas on black (not the Phase 8-9 physical render, which
ray-traces the gas through the hole's light bending; this is an artistic look at the particles).

    uv run python scripts/cinematic_preview.py RUN [sub=2] [still=DAY]
    uv run python scripts/cinematic_preview.py RUN look=clean [fps=60] [sub=4]   (for sharing)

Brightness follows the gas column (mass per pixel, log scaled, with a soft bloom); colour follows
the mass-weighted temperature (dull red at 10^4 K through orange and white to blue-white at
10^6 K). The camera path is planned over the whole run and smoothed (camera_path), and
frames between snapshots are blended particle by particle so the motion is fluid. Writes
outputs/checks/<RUN>_cinematic.mp4, and with still=DAY a 1920x1080 frame at that day,
outputs/checks/<RUN>_cinematic_dayDAY.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom
from scipy.spatial import cKDTree

from nebula import units
from nebula.analysis.cloud_metrics import snapshot_temperature
from nebula.physics.gravity import gn_dH_dp, gn_dH_dx, newton_accel
from nebula.render.video import write_video

ROOT = Path(__file__).resolve().parents[1]
FRAME_R = 12_000.0
CAM_SIGMA = 3.0  # camera smoothing, in snapshots
# temperature palette (log10 K, stretched toward the 1-10x10^4 K most gas has): ember red ->
# orange -> pale yellow -> white -> blue-white
PAL_T = np.array([4.0, 4.25, 4.55, 4.9, 5.4, 6.2])
PAL_C = np.array([[0.55, 0.07, 0.02], [0.95, 0.30, 0.05], [1.00, 0.62, 0.20],
                  [1.00, 0.88, 0.62], [0.98, 0.97, 0.95], [0.70, 0.82, 1.00]])  # fmt: skip


def _font(size):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _stars(w, h, seed=7):
    rng = np.random.default_rng(seed)
    sky = np.zeros((h, w, 3))
    n = int(w * h / 900)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    b = rng.pareto(2.2, n) * 0.05
    tint = rng.uniform(0.85, 1.0, (n, 3))
    sky[ys, xs] = np.minimum(b, 0.6)[:, None] * tint
    return gaussian_filter(sky, (0.6, 0.6, 0))


def _splat(field, hp_pix, x_pix, y_pix, w, h, s):
    """Sum `field` onto the image with each particle spread over its own smoothing length, done
    in size groups (one histogram + blur per group) so it stays fast at 1080p. Each particle is
    shared between the two nearest groups by size, so a particle growing or shrinking changes
    its look smoothly instead of stepping (which would shimmer in a video)."""
    widths = np.minimum(s * np.sqrt(2.0) ** np.arange(12), 40.0 * s)
    q = np.clip(2.0 * np.log2(np.maximum(0.7 * hp_pix, 1e-30) / s), 0.0, len(widths) - 1.0)
    k0 = np.minimum(q.astype(int), len(widths) - 2)
    frac = q - k0
    out = np.zeros((h, w))
    for k, width in enumerate(widths):
        wt = np.where(k0 == k, 1.0 - frac, 0.0) + np.where(k0 + 1 == k, frac, 0.0)
        sel = wt > 0
        if not np.any(sel):
            continue
        # wide blobs are drawn on a coarser grid and scaled up: same look, far less work
        f = int(2 ** max(0, np.floor(np.log2(width / 3.0))))
        hw, hh = -(-w // f), -(-h // f)
        wts = field[sel] * wt[sel]
        g, _, _ = np.histogram2d(x_pix[sel] / f, y_pix[sel] / f, bins=(hw, hh),
                                 range=((0, hw), (0, hh)), weights=wts)  # fmt: skip
        g = gaussian_filter(g.T[::-1], width / f)
        if f > 1:
            g = zoom(g, f, order=1)[hh * f - h :, :w] / f**2
        out += g
    return out


def _bloom(img, s):
    """Wide soft glow, computed at quarter resolution (wide blurs are slow at full size)."""
    h, w = img.shape
    small = img[: h - h % 4, : w - w % 4].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))
    blo = gaussian_filter(small, 9.0 * s / 4) + 0.5 * gaussian_filter(small, 30.0 * s / 4)
    out = zoom(blo, (h / blo.shape[0], w / blo.shape[1]), order=1)
    return out[:h, :w]


def render(x, logt, m, centre, half, w, h, stars, hp=None, ref_state=None, floor=0.004):
    """(h, w, 3) float image of the gas within the camera window.

    With `hp` (smoothing lengths) each particle is drawn as a soft blob of its own size, so the
    gas looks continuous rather than grainy. With `ref_state` (a dict carried between frames) the
    brightness scale changes slowly instead of being reset every frame, which removes flicker.
    """
    s = max(w / 1280, 1.0)
    if hp is None:
        xlo, xhi = centre[0] - half * w / h, centre[0] + half * w / h
        ylo, yhi = centre[1] - half, centre[1] + half
        rng = ((xlo, xhi), (ylo, yhi))
        mass, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=(w, h), range=rng, weights=m)
        heat, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=(w, h), range=rng, weights=m * logt)
        mass, heat = mass.T[::-1], heat.T[::-1]
        core = gaussian_filter(mass, 1.2 * s)
        tsum = gaussian_filter(heat, 1.2 * s)
    else:
        scale = h / (2 * half)
        xp = (x[:, 0] - centre[0]) * scale + w / 2
        yp = (x[:, 1] - centre[1]) * scale + h / 2
        hpp = hp * scale
        keep = (xp > -3 * hpp) & (xp < w + 3 * hpp) & (yp > -3 * hpp) & (yp < h + 3 * hpp)
        xp, yp, hpp, mk, lk = xp[keep], yp[keep], hpp[keep], m[keep], logt[keep]
        core = _splat(mk, hpp, xp, yp, w, h, s)
        tsum = _splat(mk * lk, hpp, xp, yp, w, h, s)
    bloom, tbloom = _bloom(core, s), _bloom(tsum, s)
    lum = core + 0.35 * bloom
    # colour from the same blend as the brightness, so the glow takes the colour of its gas
    tmap = (tsum + 0.35 * tbloom) / np.maximum(lum, 1e-300)
    tmap = np.where(lum > 1e-9 * max(lum.max(), 1e-300), tmap, 4.0)
    ref = np.percentile(lum[lum > 0], 99.7) if np.any(lum > 0) else 1.0
    if ref_state is not None:
        if "ref" in ref_state:
            ref = float(np.exp(0.92 * np.log(ref_state["ref"]) + 0.08 * np.log(ref)))
        ref_state["ref"] = ref
    bright = np.log1p(lum / (floor * ref)) / np.log1p(1.0 / floor)
    bright = np.clip(bright, 0, 1.15)
    colour = np.stack([np.interp(tmap, PAL_T, PAL_C[:, k]) for k in range(3)], axis=-1)
    img = stars + colour * bright[..., None] ** 1.15
    return np.clip(img, 0, 1)


def _hole(draw, centre, half, w, h):
    px = (w / 2 + (0 - centre[0]) / half * h / 2, h / 2 - (0 - centre[1]) / half * h / 2)
    if not (0 <= px[0] < w and 0 <= px[1] < h):
        return
    r = max(4, int(h / 180))
    draw.ellipse([px[0] - 2.2 * r, px[1] - 2.2 * r, px[0] + 2.2 * r, px[1] + 2.2 * r],
                 outline=(150, 190, 255), width=max(1, r // 3))  # fmt: skip
    draw.ellipse([px[0] - r, px[1] - r, px[0] + r, px[1] + r], fill=(0, 0, 0))


def _target(x):
    """Raw camera centre and half-height that frame the gas near the hole."""
    inner = x[np.linalg.norm(x, axis=1) < FRAME_R]
    inner = inner if len(inner) > 100 else x
    lo, hi = np.percentile(inner[:, :2], 2, axis=0), np.percentile(inner[:, :2], 98, axis=0)
    return 0.5 * (lo + hi), max(0.62 * float(np.max(hi - lo)), 40.0)


def drift(x, p, dt, gravity="gn", r_acc=4.0):
    """Move particles along their orbits around the hole for a time dt (negative: backwards),
    ignoring gas pressure, for drawing frames between snapshots. Straight-line blending cuts
    across the curve of gas swinging round the hole, so in-between frames would jump back and
    forth; following the orbit does not. RK4 on Hamilton's equations with the run's own gravity
    (physics.gravity: GN dx/dt = dH/dp, dp/dt = -dH/dx; Newton dx/dt = p, dp/dt = -x/r^3), with
    each particle taking enough sub-steps for its distance from the hole.
    Returns (x, inside): inside is False for particles that crossed r_acc on the way."""
    if gravity == "gn":

        def rhs(xx, pp):
            return gn_dH_dp(xx, pp), -gn_dH_dx(xx, pp)
    else:

        def rhs(xx, pp):
            return pp, newton_accel(xx)

    x, p = x.copy(), p.copy()
    r = np.linalg.norm(x, axis=1)
    n = np.clip(np.ceil(abs(dt) / (0.02 * np.maximum(r, r_acc) ** 1.5)), 1, 4000).astype(int)
    h = (dt / n)[:, None]
    inside = r > r_acc
    for k in range(int(n.max())):
        m = (k < n) & inside
        if not np.any(m):
            break
        xm, pm, hm = x[m], p[m], h[m]
        k1x, k1p = rhs(xm, pm)
        k2x, k2p = rhs(xm + 0.5 * hm * k1x, pm + 0.5 * hm * k1p)
        k3x, k3p = rhs(xm + 0.5 * hm * k2x, pm + 0.5 * hm * k2p)
        k4x, k4p = rhs(xm + hm * k3x, pm + hm * k3p)
        x[m] = xm + hm / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        p[m] = pm + hm / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        inside[m] = np.linalg.norm(x[m], axis=1) > r_acc
    return x, inside


def between(xa, pa, xb, pb, a, dt, gravity="gn", r_acc=4.0):
    """Positions a fraction `a` of the way between two snapshots, for particles in both.

    Each particle is moved forward along its orbit from the first snapshot and backward from
    the second, and the two guesses are blended. Where gas pressure rather than gravity steers a
    particle (hot gas near the hole) the two guesses disagree; there it is eased over to the
    straight-line blend, which stays between the two snapshot positions. The easing depends
    smoothly on the disagreement, so it never makes a particle jump between frames.
    Returns (x, inside): inside is False where an orbit guess crossed r_acc."""
    x1, k1 = drift(xa, pa, np.full(len(xa), a * dt), gravity, r_acc)
    x2, k2 = drift(xb, pb, np.full(len(xb), -(1 - a) * dt), gravity, r_acc)
    orbit = (1 - a) * x1 + a * x2
    line = (1 - a) * xa + a * xb
    scale = np.minimum(np.linalg.norm(xa, axis=1), np.linalg.norm(xb, axis=1))
    gap = np.linalg.norm(x1 - x2, axis=1) / scale
    trust = np.clip((0.3 - gap) / 0.2, 0.0, 1.0)[:, None]  # 1 below gap 0.1, 0 above 0.3
    trust = np.where(np.isfinite(trust), trust, 0.0)
    orbit = np.where(np.isfinite(orbit), orbit, line)
    inside = (k1 & k2) | (trust[:, 0] == 0)
    return trust * orbit + (1 - trust) * line, inside


def interpolate(pair, a, gravity, r_acc):
    """Gas a fraction `a` of the way from one snapshot to the next: each particle is moved
    forward along its orbit from the first snapshot and backward from the second (``drift``),
    and the two are blended (weight 1 - a, a), so a = 0 and a = 1 give the snapshots exactly.
    Particles only in the first snapshot (swallowed or removed before the second) follow their
    orbit until they cross r_acc; particles that reach r_acc on either path are dropped."""
    dt, ids0, x0, p0, lt0, hp0, ids1, x1, p1, lt1, hp1 = pair
    pos = {k: j for j, k in enumerate(ids1)}
    j = np.array([pos.get(k, -1) for k in ids0])
    ok = j >= 0
    xm, km = between(x0[ok], p0[ok], x1[j[ok]], p1[j[ok]], a, dt, gravity, r_acc)
    x, keep = drift(x0, p0, np.full(len(x0), a * dt), gravity, r_acc)
    x[ok], keep[ok] = xm, km
    lt, hp = lt0.copy(), hp0.copy()
    lt[ok] = (1 - a) * lt0[ok] + a * lt1[j[ok]]
    hp[ok] = np.exp((1 - a) * np.log(hp0[ok]) + a * np.log(hp1[j[ok]]))
    return x[keep], lt[keep], hp[keep]


def _even_files(files):
    """Drop snapshots saved almost on top of the one before (e.g. where a run resumed from a
    checkpoint): played one per step, they would make the motion stall for a moment."""
    t = np.array([float(np.load(f)["t"]) for f in files])
    keep = [0]
    for i in range(1, len(files)):
        dt = t[i] - t[keep[-1]]
        nearby = np.diff(t[max(i - 3, 0) : i + 4])
        if dt >= 0.35 * np.median(nearby) or i == len(files) - 1:
            keep.append(i)
    return [files[i] for i in keep]


def camera_path(files):
    """Smooth camera (centre, half) per snapshot, planned over the whole run.

    A follow-camera that reacts frame by frame either lags the fast early cloud out of view or,
    with a snap-to rule, lurches when it flips between snapping and following. Planning the path
    in advance and smoothing it forwards and backwards in time avoids both.
    """
    raw = [_target(np.load(f)["x"].astype(float)) for f in files]
    c_raw = np.array([c for c, _ in raw])
    h_raw = np.array([h for _, h in raw])
    c = gaussian_filter1d(c_raw, CAM_SIGMA, axis=0, mode="nearest")
    logh = gaussian_filter1d(np.log(h_raw), CAM_SIGMA, mode="nearest")
    # keep the gas in view where the smoothed centre cuts a corner of the raw path
    need = np.abs(c - c_raw).max(axis=1) + h_raw
    half = np.exp(gaussian_filter1d(np.maximum(logh, np.log(need)), 2.0, mode="nearest"))
    return c, half


def _decorate(img, t_day, n, centre, half, w, h, fonts):
    big, small = fonts
    im = Image.fromarray((img ** (1 / 1.1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    _hole(dr, centre, half, w, h)
    pad = int(h / 30)
    dr.text((pad, pad), f"Day {t_day:,.0f}", font=big, fill=(235, 235, 240))
    dr.text((pad, pad + int(h / 18)), "A gas cloud falling onto a black hole of 4.3 million "
            "suns", font=small, fill=(170, 175, 185))  # fmt: skip
    width_au = 2 * half * w / h * 0.0424
    dr.text((pad, h - pad - int(h / 30)), f"View {width_au:,.0f} AU across · {n:,} gas "
            "particles · colour = temperature", font=small, fill=(150, 155, 165))  # fmt: skip
    return np.asarray(im)


def _decorate_clean(img, fade):
    """The version for sharing: just the gas, no text or markers, fading in and out."""
    return (np.clip(img * fade, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8)


def _load(f, eos):
    d = np.load(f)
    logt = np.log10(snapshot_temperature(eos, d["rho"].astype(float), d["u"].astype(float)))
    x = d["x"].astype(float)
    # the simulation caps h (h_max), so sparse outflow gas would be drawn as separate blobs;
    # for display only, size each particle by the distance to its ~32nd neighbour instead
    dist, _ = cKDTree(x).query(x, k=min(33, len(x)))
    hp = np.maximum(d["h"].astype(float), 0.5 * dist[:, -1])
    return float(d["t"]), d["ids"], x, logt, hp, d["p"].astype(float)


def frames(run, sub, w=1280, h=720, only_day=None, clean=False, fps=30):
    """Video frames; `sub` frames per snapshot, the extra ones blended toward the next snapshot
    (particles matched by id; gas swallowed or removed before the next snapshot stays put)."""
    files = _even_files(sorted((ROOT / "runs" / run / "snapshots").glob("snap_*.npz")))
    manifest = json.loads((ROOT / "runs" / run / "manifest.json").read_text())
    eos = manifest["config"].get("eos", "ideal")
    gravity = manifest["config"].get("gravity", "gn")
    r_acc = float(manifest["config"].get("r_acc", 4.0))
    day = units.DAY / units.BlackHoleUnits().time_s
    stars = _stars(w, h)
    fonts = _font(int(h / 24)), _font(int(h / 42))
    if only_day is not None:
        for f in files:
            t, _, x, logt, _, _ = _load(f, eos)
            if t / day >= only_day:
                centre, half = _target(x)
                img = render(x, logt, np.ones(len(x)), centre, half, w, h, stars)
                yield _decorate(img, t / day, len(x), centre, half, w, h, fonts)
                return
        return
    cam_c, cam_h = camera_path(files)
    ref_state = {} if clean else None
    n_total, n_frame = len(files) * sub, 0
    nxt = _load(files[0], eos)
    for i in range(len(files)):
        t0, ids0, x0, lt0, hp0, p0 = nxt
        last = i + 1 == len(files)
        if not last:
            nxt = _load(files[i + 1], eos)
            t1, ids1, x1, lt1, hp1, p1 = nxt
            nxt_pair = (t1 - t0, ids0, x0, p0, lt0, hp0, ids1, x1, p1, lt1, hp1)
        for k in range(sub):
            a = k / sub
            if last or a == 0:
                x, lt, hp, t, c, hh = x0, lt0, hp0, t0, cam_c[i], cam_h[i]
            else:
                x, lt, hp = interpolate(nxt_pair, a, gravity, r_acc)
                t = (1 - a) * t0 + a * t1
                c = (1 - a) * cam_c[i] + a * cam_c[i + 1]
                hh = float(np.exp((1 - a) * np.log(cam_h[i]) + a * np.log(cam_h[i + 1])))
            if not clean:
                img = render(x, lt, np.ones(len(x)), c, hh, w, h, stars)
                yield _decorate(img, t / day, len(x), c, hh, w, h, fonts)
                continue
            img = render(
                x, lt, np.ones(len(x)), c, hh, w, h, stars, hp=hp, ref_state=ref_state, floor=0.01
            )
            sec, left = n_frame / fps, (n_total - 1 - n_frame) / fps
            fade = min(1.0, sec / 1.5, left / 2.0 + 0.0)
            n_frame += 1
            yield _decorate_clean(img, max(fade, 0.0))


def main():
    run = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    if "still" in opts:
        day = float(opts["still"])
        frame = next(frames(run, 1, 1920, 1080, only_day=day))
        out = ROOT / "outputs" / "checks" / f"{run}_cinematic_day{int(day)}.png"
        Image.fromarray(frame).save(out)
        print(f"saved {out}")
        return
    if opts.get("look") == "clean":
        # the version for sharing: 1080p, soft particles, 60 fps, no text, fades
        fps, sub = int(opts.get("fps", 60)), int(opts.get("sub", 4))
        out = ROOT / "outputs" / "checks" / f"{run}_cinematic_clean.mp4"
        gen = frames(run, sub, 1920, 1080, clean=True, fps=fps)
        if "test" in opts:  # render only frames [a, b) to check the look quickly
            lo, hi = (int(v) for v in opts["test"].split(":"))
            for k, fr in enumerate(gen):
                if k >= hi:
                    break
                if k >= lo and (k - lo) % 8 == 0:
                    Image.fromarray(fr).save(ROOT / "outputs" / "checks" / f"{run}_clean_f{k}.png")
            return
        n = write_video(gen, out, fps=fps)
        print(f"saved {out} ({n} frames)")
        return
    out = ROOT / "outputs" / "checks" / f"{run}_cinematic.mp4"
    n = write_video(frames(run, int(opts.get("sub", 2))), out)
    print(f"saved {out} ({n} frames)")


if __name__ == "__main__":
    main()
