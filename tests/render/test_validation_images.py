"""Step 2.5: shadow and Einstein-ring sizes measured in rendered images (expected-values §3):
each within 1 pixel (phases/phase-02-blackhole-render.md Step 2.5)."""

import pytest

from nebula.render.starfield import StarFieldConfig, build_cube_map, upload_cube_map
from nebula.render.validation import measure_rings, measure_shadow
from tests.helpers import check


@pytest.fixture(scope="module")
def sky():
    return upload_cube_map(build_cube_map(StarFieldConfig(n_stars=20_000, face_size=512)))


def test_shadow_size_in_image(sky):
    _, (radius, offset), _ = measure_shadow(sky)
    check(
        f"§3 shadow D=50: radius (px; {radius.expected_deg:.6f} deg)",
        radius.expected_px,
        radius.measured_px,
        atol=1.0,
    )
    check("shadow centred on the hole (px)", 0.0, offset.measured_px, atol=1.0)


def test_einstein_rings_in_image():
    _, (primary, secondary) = measure_rings()
    check(
        f"§3 primary ring D=50 (px; {primary.expected_deg:.6f} deg)",
        primary.expected_px,
        primary.measured_px,
        atol=1.0,
    )
    check(
        f"§3 secondary ring D=50 (px; {secondary.expected_deg:.6f} deg)",
        secondary.expected_px,
        secondary.measured_px,
        atol=1.0,
    )
