"""Colour swatch sheet of the nebula emission model (Phase 14 N1; outputs/checks/
nebula_colour_swatches.png): what a uniform slab of gas looks like against temperature for a few
tag mixes, with the renderer's own sources: the blackbody (source B, thermal opacity, as
march.RelativisticBlackbody) plus the emission lines (physics/emission_lines.py) through the same
slab, I = B (1 - e^-tau) + j L (1 - e^-tau) / tau. Each swatch is shown at a fixed brightness
(colour only); the number under it is its true brightness Y relative to the row's brightest.

    uv run python scripts/nebula_swatches.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nebula import units
from nebula.physics import emission_lines as el
from nebula.physics.colour import blackbody_xyz_si, srgb_encode, xyz_to_linear_srgb
from nebula.physics.opacity import thermal_absorption_code, thin_emission_fraction
from nebula.render.colour_data import load_cmf, load_emission_tables

ROOT = Path(__file__).resolve().parents[1]
lam, cmf = load_cmf()
tab = load_emission_tables()
gu = units.GasUnits()
T = np.geomspace(5e3, 1e6, 13)
MIXES = [("hydrogen only", 1.0, 0.0, 0.0), ("+ N/S (pink)", 1.0, 0.0, 0.3),
         ("+ 2% O++", 1.0, 0.02, 0.3), ("+ 5% O++ (teal accent)", 1.0, 0.05, 0.3),
         ("+ 20% O++", 1.0, 0.2, 0.3)]  # fmt: skip
# Act 1 nebula: n_H ~ 10^3 cm^-3 over 0.1 pc; Act 2 cloud: its mean density over 50 units
N_H_ACT1 = 1e9  # m^-3
RHO_ACT1 = N_H_ACT1 * units.M_P / units.X_HYDROGEN / gu.density_to_si(1.0)
PC = units.PARSEC / gu.bh.length_m  # code lengths
RHO_ACT2 = 1.0 / (4 / 3 * np.pi * 100.0**3)  # cloud mass 1 in radius 100 (sim/cloud.py)
# (title, rho, length, thin-gas continuum: S = sqrt(q/(1+q)) B instead of B; DECISIONS 2026-11-23)
SLABS = [
    ("Act 1 nebula (n_H 10^3 cm^-3, 0.1 pc; thin-gas continuum)", RHO_ACT1, 0.1 * PC, True),
    ("Act 2 cloud (its mean density, 50 units; today's continuum)", RHO_ACT2, 50.0, False),
]


def slab_xyz(rho, length, t, x_h, x_o3, x_ns, thin):
    alpha = float(thermal_absorption_code(rho, t))
    tau = alpha * length
    frac = -np.expm1(-tau) / tau if tau > 0 else 1.0
    s_fac = float(thin_emission_fraction(rho, t)) if thin else 1.0
    bb = s_fac * blackbody_xyz_si(t, lam, cmf) * -np.expm1(-tau)
    j = el.line_emission_code(rho, t, x_h, x_o3, x_ns, tab, n_cap=1e3)
    return bb + el.lines_to_xyz(j, 1.0, lam, cmf) * length * frac, tau


def main():
    fig, axes = plt.subplots(len(SLABS), 1, figsize=(15, 9.5))
    for ax, (title, rho, length, thin) in zip(axes, SLABS, strict=True):
        img = np.zeros((len(MIXES), len(T), 3))
        ys = np.zeros((len(MIXES), len(T)))
        taus = []
        for i, (_, xh, xo, xns) in enumerate(MIXES):
            for k, t in enumerate(T):
                xyz, tau = slab_xyz(rho, length, t, xh, xo, xns, thin)
                rgb = np.clip(xyz_to_linear_srgb(xyz), 0.0, None)
                img[i, k] = srgb_encode(np.clip(rgb / max(rgb.max(), 1e-300), 0.0, 1.0))
                ys[i, k] = xyz[1]
                if i == 0:
                    taus.append(tau)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        for i in range(len(MIXES)):
            for k in range(len(T)):
                ax.text(k, i + 0.38, f"{ys[i, k] / ys[i].max():.0e}", ha="center", fontsize=6.5,
                        color="w")  # fmt: skip
        ax.set_yticks(range(len(MIXES)), [m[0] for m in MIXES], fontsize=9)
        ax.set_xticks(range(len(T)), [f"{t:.0e} K\ntau {tau:.1e}" for t, tau in
                                       zip(T, taus, strict=True)], fontsize=7)  # fmt: skip
        ax.set_title(title + ": colour at fixed brightness (number = true brightness / row max)",
                     fontsize=10)  # fmt: skip
    fig.suptitle("Nebula colour model (Phase 14 N1): lines (hydrogen pink-magenta, [N II] and "
                 "[S II] red, [O III] teal) + blackbody; lines fade above ~5e4 K",
                 fontsize=11)  # fmt: skip
    fig.tight_layout()
    out = ROOT / "outputs/checks/nebula_colour_swatches.png"
    fig.savefig(out, dpi=130)
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
