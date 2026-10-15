"""Key scenario numbers in code and physical units (Phase 1, Step 1.1).

Writes outputs/checks/units.txt. Code-unit values are from PLAN.md §3 and
reference/expected-values.md (§1, §2, §2b, §5); this script only converts them.

Run: uv run python scripts/units_table.py
"""

from pathlib import Path

import numpy as np

from nebula import units
from nebula.units import BlackHoleUnits, GasUnits

OUT = Path(__file__).resolve().parents[1] / "outputs" / "checks" / "units.txt"
BH = BlackHoleUnits()
GAS = GasUnits(BH)

LENGTHS = [
    ("Length unit GM/c^2", 1.0),
    ("Event horizon (r = 2)", 2.0),
    ("Innermost stable orbit (r = 6)", 6.0),
    ("Closest approach, pericentre", 25.0),
    ("Ring forms at (circularization radius)", 50.8896237),
    ("Cloud radius", 100.0),
    ("Streams first cross (r_SI, §2b)", 1663.76),
    ("Cloud start, apocentre", 5000.0),
]
TIMES = [
    ("Time unit GM/c^3", 1.0),
    ("Ring orbit period (GN, §2)", 2303.257),
    ("Torus drain, half the mass (estimate, §5)", 26_567.0),
    ("Start to first pericentre (§2)", 396_223.627),
    ("One full radial orbit (§2)", 792_447.254),
    ("Third pericentre passage, mean (§2b, likely first crash)", 1_979_500.0),
]


def code(x: float) -> str:
    return f"{x:,.0f}" if x >= 1e4 else f"{x:,.6g}"


def main() -> None:
    lines = [
        f"Black hole {BH.mass_msun / 1e6:.3g} million M_sun (like Sgr A*), "
        f"cloud {GAS.cloud_mass_msun} M_sun",
        "Code units: G = c = M = 1.",
        "",
        f"{'Length':58s} {'code':>12s}   physical",
    ]
    lines += [
        f"  {n:56s} {code(r):>12s}   {units.format_length(BH.length_to_m(r))}" for n, r in LENGTHS
    ]
    lines += ["", f"{'Time':58s} {'code':>12s}   physical"]
    lines += [f"  {n:56s} {code(t):>12s}   {units.format_time(BH.time_to_s(t))}" for n, t in TIMES]
    rho = GAS.density_to_si(1.0 / (4.0 / 3.0 * np.pi * 100.0**3))
    cs = units.gas_sound_speed_from_temperature(1e4)
    lines += [
        "",
        "Cloud gas",
        f"  {'Temperature':56s} 10,000 K",
        f"  {'Density (uniform sphere)':56s} {rho:.4g} kg/m^3",
        f"  {'Sound speed':56s} {cs * units.C / units.KM:.4g} km/s = {cs:.4g} c",
        f"  {'Orbital speed at start (§2)':56s} "
        f"{1.4701605691e-3 * units.C / units.KM:.4g} km/s = 1.470e-3 c",
    ]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(text)
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
