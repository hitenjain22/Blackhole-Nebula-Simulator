"""Rule 2 check for our Phantom patch (phantom/nebula_phantom.patch): Phantom's hydro
acceleration and du/dt on the same particles as our Python code (sim/hydro.evaluate, whose
Numba kernels are tested against the readable reference in tests/sim/).

    uv run --with sarracen --with scikit-learn python scripts/phantom_force_check.py \
        DUMP_RECON_ON DUMP_RECON_OFF [n_sample=4000] [pressure=phantom] [input=PHANTOM_INPUT_DUMP]
        [save=tests/data/phantom_force_sample.npz]

The dumps come from one derivative evaluation in Phantom (nebula_dumpderivs = T, cooling off,
nebula_cond = 1, nebula_hmax = 200; nebula_recon 1 and 0) on the same state. Our code gets
Phantom's positions, velocities, u and viscosity alpha, solves h and density itself, and
evaluates the same forces. input=: the dump Phantom started from, for the alpha it used. Masses are converted from Phantom's M_bh to our gas mass unit
(PLAN §3; u, velocities and lengths are in the same code units, G = c = M_bh = 1).
Prints the median and 99th-percentile relative differences; writes
outputs/checks/phantom_force_check.png.
"""

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sarracen  # noqa: E402

from nebula import units  # noqa: E402
from nebula.sim.coupled import CoupledConfig  # noqa: E402
from nebula.sim.hydro import HydroState, eos_state, evaluate  # noqa: E402
from nebula.sim.neighbours import tighten  # noqa: E402
from nebula.sim.sph import compute_density, forces_numba, velocity_gradients_numba  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    s = sarracen.read_phantom(str(path))
    return s[0] if isinstance(s, tuple) else s


def alpha_used(s, inp):
    """The viscosity alpha Phantom used in its start-up force pass: the input dump's (Phantom
    raises alpha to max(alpha, alpha_loc) only after that pass, initial.F90), matched by iorig."""
    if inp is None:
        return s["alpha"].values.astype(float)
    a = dict(zip(inp["iorig"].values, inp["alpha"].values.astype(float)))
    return np.array([a[i] for i in s["iorig"].values])


def ours(s, recon: bool):
    hc = replace(CoupledConfig().hydro, eos="gasrad", reconstruction=recon)
    m_ph = float(s.params["massoftype"])
    m = m_ph * units.BlackHoleUnits().mass_kg / hc.gas_units.mass_kg_per_code
    n = len(s)
    st = HydroState(
        0.0, s[["x", "y", "z"]].values.astype(float), s[["vx", "vy", "vz"]].values.astype(float),
        s["u"].values.astype(float), np.full(n, m), s["h"].values.astype(float),
        s["alpha_used"].values,
    )  # fmt: skip
    return evaluate(st, hc), m_ph / m


def ours_phantom_pressure(s, recon: bool, ev, mratio):
    """Our force formulas with Phantom's pressure (and our sound speed scaled to it): isolates
    the force/viscosity/conductivity formulas from the equation of state, whose physical
    constants differ slightly between the codes (Phantom physcon.f90 vs units.py CODATA)."""
    hc = replace(CoupledConfig().hydro, eos="gasrad", reconstruction=recon)
    x = s[["x", "y", "z"]].values.astype(float)
    v = s[["vx", "vy", "vz"]].values.astype(float)
    d = compute_density(x, ev.mass, ev.h, kernel_name=hc.kernel, eta=hc.eta, h_max=hc.h_max)
    p_ours, cs_ours = eos_state(hc, d.rho, s["u"].values.astype(float))
    p = s["pressure"].values.astype(float) / mratio
    cs = cs_ours * np.sqrt(p / p_ours)
    nb = tighten(d.neighbours, x, d.h)
    shock = hc.shock()
    gv = velocity_gradients_numba(x, v, ev.mass, d.h, nb, hc.kernel) if recon else None
    f = forces_numba(x, v, ev.mass, d.h, d.rho, d.omega, p, nb, hc.kernel,
                     u=s["u"].values.astype(float), c_s=cs, alpha=s["alpha_used"].values,
                     shock=shock, grad_v=gv)  # fmt: skip
    return f.acc, f.dudt


def rel(a, b, scale):
    return np.abs(a - b) / scale


def save_sample(on, off, path, center=(0.0, 0.0, 0.0), r_in=600.0, margin=1200.0):
    """Particles within r_in + margin of center, with Phantom's results for both cases, for
    tests/sim/test_phantom_patch.py. Only those within r_in are compared there: their
    neighbours (r_in + 400), the neighbours' gradients and densities (r_in + 800, + 1200) are
    then complete (2 h_max = 400)."""
    assert np.array_equal(on["iorig"].values, off["iorig"].values)
    x = on[["x", "y", "z"]].values.astype(float)
    keep = np.linalg.norm(x - np.asarray(center), axis=1) < r_in + margin
    col = lambda s, c: s[c].values[keep].astype(float)  # noqa: E731
    np.savez_compressed(
        path, x=x[keep], v=on[["vx", "vy", "vz"]].values[keep].astype(float),
        u=col(on, "u"), h_phantom=col(on, "h"), alpha=on["alpha_used"].values[keep],
        pressure=col(on, "pressure"),
        acc_on=on[["fx", "fy", "fz"]].values[keep].astype(float), dudt_on=col(on, "dudt"),
        acc_off=off[["fx", "fy", "fz"]].values[keep].astype(float), dudt_off=col(off, "dudt"),
        mass_phantom=float(on.params["massoftype"]), center=np.asarray(center, float),
        r_in=r_in, time=float(on.params["time"]),
    )  # fmt: skip
    print(f"saved {path}: {keep.sum()} particles, "
          f"{(np.linalg.norm(x[keep] - np.asarray(center), axis=1) < r_in).sum()} compared")


def main():
    on, off = load(sys.argv[1]), load(sys.argv[2])
    opts = dict(a.split("=", 1) for a in sys.argv[3:])
    inp = load(opts["input"]) if "input" in opts else None
    for s in (on, off):
        s["alpha_used"] = alpha_used(s, inp)
    if "save" in opts:
        save_sample(on, off, opts["save"])
        return
    nsamp = int(opts.get("n_sample", 4000))
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for k, (name, s, recon) in enumerate((("recon on", on, True), ("recon off", off, False))):
        ev, mratio = ours(s, recon)
        a_ph = s[["fx", "fy", "fz"]].values.astype(float)
        du_ph = s["dudt"].values.astype(float)
        if opts.get("pressure") == "phantom":
            acc, dudt = ours_phantom_pressure(s, recon, ev, mratio)
            ev = replace(ev, acc=acc, dudt=dudt)
            name += ", Phantom P"
        amag = np.linalg.norm(ev.acc, axis=1)
        # relative to each particle's own scale: |a| and |du/dt| (plus a tiny floor for the
        # particles with almost no force)
        da = np.linalg.norm(ev.acc - a_ph, axis=1) / np.maximum(amag, 1e-6 * np.median(amag))
        ddu = np.abs(ev.dudt - du_ph) / np.maximum(np.abs(ev.dudt),
                                                   1e-6 * np.median(np.abs(ev.dudt)))
        dh = np.abs(ev.h - s["h"].values) / ev.h
        rows.append((name, np.median(dh), np.percentile(dh, 99), np.median(da),
                     np.percentile(da, 99), np.median(ddu), np.percentile(ddu, 99)))  # fmt: skip
        rng = np.random.default_rng(0)
        i = rng.choice(len(s), min(nsamp, len(s)), replace=False)
        ax = axes[0, k]
        ax.loglog(amag[i], np.linalg.norm(a_ph[i], axis=1), ".", ms=2)
        lim = [amag[i].min(), amag[i].max()]
        ax.plot(lim, lim, "k-", lw=0.6)
        ax.set_xlabel("|a_hydro|, our code"); ax.set_ylabel("|a_hydro|, Phantom")  # fmt: skip
        ax.set_title(f"{name}: acceleration (each dot a particle)")
        bx = axes[1, k]
        x, y = ev.dudt[i], du_ph[i]
        bx.plot(x, y, ".", ms=2)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        bx.plot(lim, lim, "k-", lw=0.6)
        bx.set_xscale("symlog", linthresh=1e-14); bx.set_yscale("symlog", linthresh=1e-14)  # fmt: skip
        bx.set_xlabel("du/dt, our code"); bx.set_ylabel("du/dt, Phantom")  # fmt: skip
        bx.set_title(f"{name}: heating/cooling rate du/dt")
    fig.suptitle("Same particles (Phantom day 300): Phantom with our patch vs our code")
    fig.tight_layout()
    tag = "_phantomP" if opts.get("pressure") == "phantom" else ""
    out = ROOT / "outputs" / "checks" / f"phantom_force_check{tag}.png"
    fig.savefig(out, dpi=90)
    print(f"{'case':10s} {'h med':>9s} {'h p99':>9s} {'a med':>9s} {'a p99':>9s} "
          f"{'du med':>9s} {'du p99':>9s}")  # fmt: skip
    for r in rows:
        print(f"{r[0]:10s} " + " ".join(f"{v:9.2e}" for v in r[1:]))
    print(out)


if __name__ == "__main__":
    main()
