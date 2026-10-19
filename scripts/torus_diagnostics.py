"""Phase 6 diagnostics of a run (phases/phase-06-shocks-cooling-ring.md 6.5), from its snapshots.

    uv run python scripts/torus_diagnostics.py RUN_NAME

outputs/checks/<RUN>_torus.txt   per snapshot: bound / unbound / plunging mass, ring mass (e < 0.3,
                                 r < 200), its radius, H/r and extent, mass in shocks and where,
                                 luminosity (W, L_Edd), optical-depth fractions
outputs/checks/<RUN>_torus.png   eccentricity histogram against time (the circularization movie in
                                 one plot), ring mass, mass in shocks, luminosity, accretion rate
Temperatures use the run's equation of state (manifest config "eos").
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nebula import units  # noqa: E402
from nebula.analysis.torus_metrics import (  # noqa: E402
    accretion_rate,
    eccentricity_histogram,
    luminosity_watts,
    mass_in_shocks,
    optical_depth_fractions,
    orbit_elements,
    ring_measures,
    unbound_fraction,
)
from nebula.sim.hydro import HydroConfig, temperature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = sys.argv[1]
    folder = ROOT / "runs" / run
    cfg = json.loads((folder / "manifest.json").read_text())["config"]
    hc = HydroConfig(eos=cfg.get("eos", "ideal"))
    files = sorted((folder / "snapshots").glob("snap_*.npz"))
    n0 = len(np.load(files[0])["x"])
    l_edd = units.eddington_luminosity_w()
    # per-snapshot results are cached (keyed by file name, size and time), so a progress check
    # of a long run only analyses the snapshots that are new since the last one
    cache_path = ROOT / "outputs" / "cache" / f"{run}_torus_rows.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows, ecc_maps = [], []
    for f in files:
        key = f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
        if key in cache:
            rows.append(cache[key]["row"])
            ecc_maps.append(np.array(cache[key]["hist"]))
            continue
        d = np.load(f)
        x, p = d["x"].astype(float), d["p"].astype(float)
        u, rho, h = d["u"].astype(float), d["rho"].astype(float), d["h"].astype(float)
        m = np.full(len(x), 1.0 / n0)
        el = orbit_elements(x, p)
        rm = ring_measures(x, m, el)
        _, hist = eccentricity_histogram(el, m)
        ecc_maps.append(hist)
        temps = temperature(hc, rho, u)
        f_es, f_eff = optical_depth_fractions(rho, temps, h, m)
        visc = d["visc_heating"].astype(float) if "visc_heating" in d.files else np.zeros(len(x))
        lum = d["luminosity"].astype(float) if "luminosity" in d.files else np.zeros(len(x))
        shock_frac, _ = mass_in_shocks(x, u, visc, m)
        l_w = float(luminosity_watts(np.sum(lum)))
        rows.append({"t": float(d["t"]), "bound": float(np.sum(m[el.kind == 0])),
                     "unbound": unbound_fraction(el, m), "plunging": float(np.sum(m[el.kind == 2])),
                     "ring": rm.mass_fraction, "ring_r": rm.mean_radius, "ring_hr": rm.thickness,
                     "ring_r10": rm.r10, "ring_r90": rm.r90, "shocks": shock_frac,
                     "lum_w": l_w, "lum_edd": l_w / l_edd, "tau_es_thin": f_es,
                     "tau_eff_thin": f_eff})  # fmt: skip
        cache[key] = {"row": rows[-1], "hist": [float(v) for v in hist]}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    lines = [f"{run}: {n0} particles; eos {hc.eos}; mass fractions of the starting cloud", "",
             "t  bound  unbound  plunging  ring(e<0.3,r<200 of bound)  ring_r  H/r  r10-r90  "
             "shocks  L(W)  L/L_Edd  tau_es,h<1  tau_eff,h<1"]  # fmt: skip
    for r in rows:
        lines.append(
            f"{r['t']:10.0f} {r['bound']:.3f} {r['unbound']:.3f} {r['plunging']:.3f} "
            f"{r['ring']:.3f} {r['ring_r']:7.1f} {r['ring_hr']:.3f} {r['ring_r10']:6.1f}-"
            f"{r['ring_r90']:6.1f} {r['shocks']:.3f} {r['lum_w']:.2e} {r['lum_edd']:.2e} "
            f"{r['tau_es_thin']:.3f} {r['tau_eff_thin']:.3f}"
        )
    removed = folder / "removed.json"
    log = json.loads(removed.read_text()) if removed.exists() else []
    acc = [e for e in log if e["reason"] == "accreted"]
    if acc:
        t_acc = np.array([e["t"] for e in acc])
        edges = np.linspace(rows[0]["t"], rows[-1]["t"], 30)
        _, rate, edd = accretion_rate(t_acc, np.array([e["m"] for e in acc]), edges)
        lines.append(f"\naccreted {len(acc)} particles; peak rate {rate.max():.3e} M_sun/yr "
                     f"({edd.max():.2f} Mdot_Edd)")  # fmt: skip
    text = "\n".join(lines)
    out = ROOT / "outputs" / "checks"
    (out / f"{run}_torus.txt").write_text(text + "\n")
    print(text)

    t = np.array([r["t"] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    em = np.array(ecc_maps).T
    axes[0, 0].pcolormesh(np.arange(len(t) + 1), np.linspace(0, 1, em.shape[0] + 1), em,
                          cmap="viridis", shading="flat")  # fmt: skip
    axes[0, 0].set_xlabel("snapshot number (time increases to the right)")
    axes[0, 0].set_ylabel("orbital eccentricity")
    axes[0, 0].set_title("Eccentricity of the gas over time (circular = 0)")
    axes[0, 1].plot(t, [r["ring"] for r in rows], label="ring (e < 0.3, r < 200), of bound mass")
    axes[0, 1].plot(t, [r["unbound"] for r in rows], label="unbound")
    axes[0, 1].plot(t, [r["shocks"] for r in rows], label="in shocks")
    axes[0, 1].set_title("Mass fractions")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].semilogy(t, np.maximum([r["lum_edd"] for r in rows], 1e-12))
    axes[1, 0].set_title("Luminosity (Eddington units)")
    axes[1, 1].plot(t, [r["tau_eff_thin"] for r in rows], label="tau_eff,h < 1")
    axes[1, 1].plot(t, [r["tau_es_thin"] for r in rows], label="tau_es,h < 1")
    axes[1, 1].set_title("Optically thin mass fractions")
    axes[1, 1].legend(fontsize=8)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("time (GM/c³)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{run}: Phase 6 diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{run}_torus.png", dpi=110)


if __name__ == "__main__":
    main()
