"""Step 5.0.1: is the pericentre collision real? (docs/phase-05-breakdown.md)

The cloud (sim/cloud.py) is moved with gravity only (no gas forces), all particles in
lockstep with the generalized leapfrog, step eta_orb * min(r)^1.5. At times around the leading
edge's pericentre we evaluate the SPH smoothing lengths the particle distribution *would* have,
and count pairs within kernel reach (r_ab < 2 max(h_a, h_b)) made of one inbound and one
outbound particle: SPH would make them interact although, moving ballistically, they never
meet. Done at 1k, 10k and 50k particles (a cheap forecast of how resolution changes this).

Output: outputs/checks/plunge_geometry.txt
Run: uv run python scripts/investigations/plunge_geometry.py
"""

from pathlib import Path

import numpy as np

from nebula.physics.gravity import gn_p_from_v, gn_v_from_p
from nebula.sim.cloud import cloud_ic
from nebula.sim.integrator import glf_step_many
from nebula.sim.neighbours import find_neighbours_numba
from nebula.sim.sph import compute_density

OUT = Path(__file__).resolve().parents[2] / "outputs" / "checks" / "plunge_geometry.txt"
TIMES = (380_000.0, 383_000.0, 385_000.0, 386_646.0, 390_000.0, 396_224.0)
ETA = 0.00125


def evolve(x, p, t_marks):
    t, out = 0.0, {}
    for t_mark in t_marks:
        while t < t_mark:
            r_min = float(np.min(np.linalg.norm(x, axis=1)))
            dt = min(ETA * r_min**1.5, t_mark - t)
            x, p, _ = glf_step_many(x, p, dt)
            t += dt
        out[t_mark] = (x.copy(), p.copy())
    return out


def leg_coupling(x, p, mass, h_guess):
    d = compute_density(x, mass, h_guess)
    h = d.h
    v = gn_v_from_p(x, p)
    r = np.linalg.norm(x, axis=1)
    vr = np.sum(x * v, axis=1) / r
    near = np.flatnonzero(r < 100)
    nb = find_neighbours_numba(x, h)
    n_pairs = n_cross = 0
    dv_cross = []
    for a in near:
        for b in nb.index[nb.offset[a] : nb.offset[a + 1]]:
            if b <= a or np.linalg.norm(x[a] - x[b]) >= 2 * max(h[a], h[b]):
                continue
            n_pairs += 1
            if vr[a] * vr[b] < 0:
                n_cross += 1
                dv_cross.append(np.linalg.norm(v[a] - v[b]))
    h_over_r = h[near] / r[near] if len(near) else np.array([np.nan])
    return {
        "n_near": len(near),
        "h_med": float(np.median(h[near])) if len(near) else np.nan,
        "h_over_r_med": float(np.median(h_over_r)),
        "pairs": n_pairs,
        "cross_pairs": n_cross,
        "dv_cross_med": float(np.median(dv_cross)) if dv_cross else 0.0,
        "h_all_med": float(np.median(h)),
    }


def main():
    lines = [
        "Gravity-only cloud; SPH smoothing lengths evaluated on the particle positions.",
        "near = particles with r < 100; cross pairs = inbound-outbound pairs within kernel reach",
        "",
    ]
    for n in (1000, 10_000, 50_000):
        ic = cloud_ic(n)
        x, p = ic.x.copy(), gn_p_from_v(ic.x, ic.v)
        states = evolve(x, p, TIMES)
        h_guess = ic.h.copy()
        for t, (xs, ps) in states.items():
            s = leg_coupling(xs, ps, ic.mass, h_guess)
            line = (
                f"N={len(ic.x):6d} t={t:9.0f}: near {s['n_near']:5d}, h median there "
                f"{s['h_med']:6.1f} (h/r {s['h_over_r_med']:.2f}; "
                f"whole cloud {s['h_all_med']:.1f}), "
                f"pairs {s['pairs']:7d}, inbound-outbound {s['cross_pairs']:6d} "
                f"(closing speed median {s['dv_cross_med']:.3f})"
            )
            print(line, flush=True)
            lines.append(line)
        lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
